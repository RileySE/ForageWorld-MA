"""
Causal, fixed-window Transformer sequence model for the PPO baselines.

This module provides `ScannedTransformer`, a **drop-in replacement for
`ScannedRNN`** in ippo_rnn.py / central_ippo_rnn.py / seperate_ippo_rnn.py.
It keeps exactly the same call contract, so the surrounding PPO machinery
(rollout scan, GAE, minibatching, checkpointing) is unchanged:

    carry, y = ScannedTransformer(...)(carry, (ins, resets))

        ins    : (T, B, d_in)   per-step embedding, same tensor the GRU got
        resets : (T, B)         `last_done`; True => this step starts a new episode
        carry  : (B, H, d_model + 1)   single array, see "Carry layout" below
        y      : (T, B, out_dim)       same width as the GRU output by default

WHY A WINDOW, NOT THE WHOLE HISTORY
-----------------------------------
The carry holds only the last `max_history` (H) token embeddings.  At step t the
model attends over timesteps [t-H+1, t] and nothing older.  Memory and compute
per step are therefore O(H), constant in episode length -- craftax episodes run
to 9000 steps, so feeding the full history is not an option.

Carry layout
------------
One float array of shape (B, H, d_model + 1) so it stays a plain tensor like the
GRU hidden state (no pytree), which is what the existing minibatch / checkpoint
code expects:

    carry[..., :d_model]  ring buffer of projected tokens, OLDEST at index 0,
                          CURRENT step at index H-1 (buffer shifts left each step)
    carry[...,  d_model]  1.0 = real token, 0.0 = padding or pre-episode-boundary

Because the buffer shifts left every step, slot i is always "(H-1-i) steps ago".
The learned positional embedding is therefore a *relative* position code by
construction -- no absolute timestep ever enters the model.

Masking
-------
Attention uses `causal AND key_is_valid`, so a query never sees (a) its own
future, (b) zero padding at the start of an episode, or (c) anything from before
an episode boundary.  The diagonal is always left open so a fully-masked row
cannot produce a NaN softmax.

Recompute semantics
-------------------
The window is recomputed from scratch at every step (no KV cache).  Rollout and
the PPO loss recomputation run the *identical* code path, so the importance
ratio is exact at the first epoch, exactly as for the GRU.  Cost: attention is
recomputed H times per token, i.e. ~H x the FLOPs of a KV-cached/banded
implementation.  A banded-mask parallel formulation over the whole rollout would
be ~H x cheaper but is NOT numerically identical for n_layers > 1 (query t-k
would see the window [t-k-H+1, t-k] instead of the window [t-H+1, t-k] the
scanned version gives it), which would silently corrupt the PPO ratio.  We keep
the exact version.

Parameter budget
----------------
`transformer_core_params` / `gru_core_params` below count the sequence-model core
(input projection + blocks + output projection) so a transformer config can be
matched to the GRU it replaces.  `resolve_d_model` solves for the d_model that
lands closest to the GRU budget.  With the defaults in DEFAULT_HPARAMS and
FC_DIM_SIZE=128 / GRU_HIDDEN_DIM=512 the transformer sits at ~103% of the GRU.
"""

import functools
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Every knob of the transformer policy, with its default.  Config keys are the
# upper-case names below; see the module docstring in the runner for the table.
DEFAULT_HPARAMS: Dict[str, Any] = {
    # -- sequence window -----------------------------------------------------
    "TF_MAX_HISTORY":     16,      # H: attention window length, in env steps.
                                   #    Also the size of the carry ring buffer.
    # -- width / depth -------------------------------------------------------
    "TF_D_MODEL":         192,     # residual-stream width. "auto" -> solve for
                                   # the value that matches the GRU param count.
    "TF_N_LAYERS":        2,       # number of transformer blocks (L)
    "TF_N_HEADS":         4,       # attention heads; must divide TF_D_MODEL
                                   # (192/4 = 48 dims per head)
    "TF_MLP_RATIO":       4.0,     # FFN hidden width = ratio * d_model (768)
    "TF_OUT_DIM":         None,    # width of y. None -> GRU_HIDDEN_DIM, so the
                                   # actor/critic/aux heads are unchanged.
    # -- block internals -----------------------------------------------------
    "TF_ACTIVATION":      "gelu",  # "gelu" | "relu" | "swish"
    "TF_NORM_PLACEMENT":  "pre",   # "pre" (pre-LN, stable) | "post"
    "TF_POS_ENCODING":    "learned",  # "learned" | "sinusoidal" | "none"
    "TF_QKV_BIAS":        True,    # bias on q/k/v/out projections
    "TF_ATTN_DROPOUT":    0.0,     # requires rngs={'dropout': ...} plumbing
    "TF_RESID_DROPOUT":   0.0,     # requires rngs={'dropout': ...} plumbing
    # -- init ----------------------------------------------------------------
    "TF_INIT_SCALE":      1.0,     # orthogonal gain for in/out projections
}

_ACTIVATIONS = {"gelu": nn.gelu, "relu": nn.relu, "swish": nn.swish, "silu": nn.swish}


def hparams_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull TF_* keys out of a run config, filling in DEFAULT_HPARAMS.

    Resolves TF_OUT_DIM (None -> GRU_HIDDEN_DIM) and TF_D_MODEL ("auto" ->
    param-matched to the GRU), then validates.  Returns a plain dict of
    lower-case kwargs for ScannedTransformer.
    """
    hp = {k: config.get(k, v) for k, v in DEFAULT_HPARAMS.items()}

    if hp["TF_OUT_DIM"] is None:
        hp["TF_OUT_DIM"] = int(config["GRU_HIDDEN_DIM"])

    d_in = int(config["FC_DIM_SIZE"])
    if isinstance(hp["TF_D_MODEL"], str):
        if hp["TF_D_MODEL"].lower() != "auto":
            raise ValueError(f'TF_D_MODEL must be an int or "auto", got {hp["TF_D_MODEL"]!r}')
        hp["TF_D_MODEL"] = resolve_d_model(
            target=gru_core_params(d_in, int(config["GRU_HIDDEN_DIM"])),
            input_dim=d_in,
            n_layers=int(hp["TF_N_LAYERS"]),
            mlp_ratio=float(hp["TF_MLP_RATIO"]),
            out_dim=int(hp["TF_OUT_DIM"]),
            max_history=int(hp["TF_MAX_HISTORY"]),
            multiple_of=int(hp["TF_N_HEADS"]),
            pos_encoding=hp["TF_POS_ENCODING"],
            qkv_bias=bool(hp["TF_QKV_BIAS"]),
        )

    d_model, n_heads = int(hp["TF_D_MODEL"]), int(hp["TF_N_HEADS"])
    if d_model % n_heads:
        raise ValueError(
            f"TF_D_MODEL ({d_model}) must be divisible by TF_N_HEADS ({n_heads})"
        )
    if int(hp["TF_MAX_HISTORY"]) < 1:
        raise ValueError(f'TF_MAX_HISTORY must be >= 1, got {hp["TF_MAX_HISTORY"]}')
    if hp["TF_ACTIVATION"] not in _ACTIVATIONS:
        raise ValueError(
            f'TF_ACTIVATION must be one of {sorted(_ACTIVATIONS)}, got {hp["TF_ACTIVATION"]!r}'
        )
    if hp["TF_NORM_PLACEMENT"] not in ("pre", "post"):
        raise ValueError(f'TF_NORM_PLACEMENT must be "pre" or "post", got {hp["TF_NORM_PLACEMENT"]!r}')
    if hp["TF_POS_ENCODING"] not in ("learned", "sinusoidal", "none"):
        raise ValueError(
            f'TF_POS_ENCODING must be "learned"|"sinusoidal"|"none", got {hp["TF_POS_ENCODING"]!r}'
        )
    if float(hp["TF_ATTN_DROPOUT"]) > 0.0 or float(hp["TF_RESID_DROPOUT"]) > 0.0:
        # The runners call network.apply without an rng dict, and PPO's ratio
        # requires the collection and update passes to be identical anyway.
        raise NotImplementedError(
            "TF_ATTN_DROPOUT / TF_RESID_DROPOUT > 0 need rngs={'dropout': ...} threaded "
            "through every network.apply call (rollout, loss, video). Not wired up; "
            "keep them at 0.0."
        )

    return {
        "d_model": d_model,
        "n_layers": int(hp["TF_N_LAYERS"]),
        "n_heads": n_heads,
        "mlp_ratio": float(hp["TF_MLP_RATIO"]),
        "max_history": int(hp["TF_MAX_HISTORY"]),
        "out_dim": int(hp["TF_OUT_DIM"]),
        "activation": hp["TF_ACTIVATION"],
        "norm_placement": hp["TF_NORM_PLACEMENT"],
        "pos_encoding": hp["TF_POS_ENCODING"],
        "qkv_bias": bool(hp["TF_QKV_BIAS"]),
        "attn_dropout": float(hp["TF_ATTN_DROPOUT"]),
        "resid_dropout": float(hp["TF_RESID_DROPOUT"]),
        "init_scale": float(hp["TF_INIT_SCALE"]),
    }


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------
def gru_core_params(input_dim: int, hidden: int) -> int:
    """Parameter count of the `ScannedRNN` core (flax GRUCell).

    NOTE: flax's GRUCell takes its hidden width from `carry.shape[-1]`, so the
    `features=ins.shape[1]` argument in ScannedRNN is inert -- the real hidden
    size is GRU_HIDDEN_DIM, not FC_DIM_SIZE.

    Layout: i{r,z,n} = (in,h)+h ; h{r,z} = (h,h) no bias ; hn = (h,h)+h.
    """
    return 3 * (input_dim * hidden + hidden) + 2 * (hidden * hidden) + (hidden * hidden + hidden)


def transformer_core_params(
    input_dim: int,
    d_model: int,
    n_layers: int,
    mlp_ratio: float,
    out_dim: int,
    max_history: int,
    pos_encoding: str = "learned",
    qkv_bias: bool = True,
) -> int:
    """Parameter count of the `ScannedTransformer` core."""
    d, r = int(d_model), int(round(mlp_ratio * d_model))
    in_proj = input_dim * d + d
    pos = max_history * d if pos_encoding == "learned" else 0
    attn = 4 * d * d + (4 * d if qkv_bias else 0)   # q,k,v,out kernels (+biases)
    mlp = (d * r + r) + (r * d + d)                 # two Dense layers with bias
    norms = 2 * (2 * d)                             # two LayerNorms (scale+bias)
    out_proj = d * out_dim + out_dim
    out_ln = 2 * d                                  # final LayerNorm on y
    return in_proj + pos + n_layers * (attn + mlp + norms) + out_ln + out_proj


def resolve_d_model(
    target: int,
    input_dim: int,
    n_layers: int,
    mlp_ratio: float,
    out_dim: int,
    max_history: int,
    multiple_of: int = 4,
    pos_encoding: str = "learned",
    qkv_bias: bool = True,
    search_max: int = 4096,
) -> int:
    """Smallest-error d_model (a multiple of `multiple_of`) hitting `target` params."""
    best, best_err = multiple_of, None
    for d in range(multiple_of, search_max + 1, multiple_of):
        err = abs(
            transformer_core_params(
                input_dim, d, n_layers, mlp_ratio, out_dim, max_history, pos_encoding, qkv_bias
            )
            - target
        )
        if best_err is None or err < best_err:
            best, best_err = d, err
        elif err > best_err:
            break  # convex in d, so the first uptick is the minimum
    return best


def describe(hp: Dict[str, Any], input_dim: int, gru_hidden: int) -> str:
    """Human-readable hyperparameter + parameter-budget report."""
    tf_p = transformer_core_params(
        input_dim, hp["d_model"], hp["n_layers"], hp["mlp_ratio"],
        hp["out_dim"], hp["max_history"], hp["pos_encoding"], hp["qkv_bias"],
    )
    gru_p = gru_core_params(input_dim, gru_hidden)
    head_dim = hp["d_model"] // hp["n_heads"]
    ffn = int(round(hp["mlp_ratio"] * hp["d_model"]))
    lines = [
        "─" * 68,
        "Transformer policy (causal, fixed-window) — sequence-model core",
        "─" * 68,
        f"  TF_MAX_HISTORY     {hp['max_history']:<10}  attention window (env steps)",
        f"  TF_D_MODEL         {hp['d_model']:<10}  residual stream width",
        f"  TF_N_LAYERS        {hp['n_layers']:<10}  transformer blocks",
        f"  TF_N_HEADS         {hp['n_heads']:<10}  heads ({head_dim} dims/head)",
        f"  TF_MLP_RATIO       {hp['mlp_ratio']:<10}  FFN hidden = {ffn}",
        f"  TF_OUT_DIM         {hp['out_dim']:<10}  output width (feeds actor/critic/aux heads)",
        f"  TF_ACTIVATION      {hp['activation']:<10}",
        f"  TF_NORM_PLACEMENT  {hp['norm_placement']:<10}  LayerNorm placement",
        f"  TF_POS_ENCODING    {hp['pos_encoding']:<10}  relative by construction (shifting window)",
        f"  TF_QKV_BIAS        {str(hp['qkv_bias']):<10}",
        f"  TF_ATTN_DROPOUT    {hp['attn_dropout']:<10}",
        f"  TF_RESID_DROPOUT   {hp['resid_dropout']:<10}",
        f"  TF_INIT_SCALE      {hp['init_scale']:<10}  orthogonal gain, in/out projections",
        "─" * 68,
        f"  input dim (FC_DIM_SIZE)      {input_dim}",
        f"  transformer core params      {tf_p:,}",
        f"  GRU core params (h={gru_hidden})     {gru_p:,}",
        f"  ratio transformer / GRU      {tf_p / gru_p:.3f}",
        f"  carry per env                {hp['max_history']} x {hp['d_model'] + 1} = "
        f"{hp['max_history'] * (hp['d_model'] + 1):,} floats"
        f"   (GRU: {gru_hidden:,})",
        "─" * 68,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _sinusoidal(max_history: int, d_model: int) -> jnp.ndarray:
    """Fixed sinusoidal code over *relative* offsets (slot 0 = oldest)."""
    pos = np.arange(max_history)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))[None, :]
    pe = np.zeros((max_history, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return jnp.asarray(pe)


class TransformerBlock(nn.Module):
    """One pre-LN (or post-LN) causal self-attention + FFN block."""
    d_model: int
    n_heads: int
    mlp_ratio: float
    activation: str
    norm_placement: str
    qkv_bias: bool
    attn_dropout: float
    resid_dropout: float

    @nn.compact
    def __call__(self, x, mask, deterministic: bool = True):
        act = _ACTIVATIONS[self.activation]
        ffn_dim = int(round(self.mlp_ratio * self.d_model))

        def attn(h):
            return nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                use_bias=self.qkv_bias,
                dropout_rate=self.attn_dropout,
                deterministic=deterministic,
            )(h, h, mask=mask)

        def mlp(h):
            h = nn.Dense(ffn_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
            h = act(h)
            return nn.Dense(
                self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(h)

        if self.norm_placement == "pre":
            x = x + attn(nn.LayerNorm()(x))
            x = x + mlp(nn.LayerNorm()(x))
        else:  # post-LN
            x = nn.LayerNorm()(x + attn(x))
            x = nn.LayerNorm()(x + mlp(x))
        return x


class _ScannedTransformerCore(nn.Module):
    """Steps the window one env step at a time; params shared across steps."""
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    max_history: int
    out_dim: int
    activation: str
    norm_placement: str
    pos_encoding: str
    qkv_bias: bool
    attn_dropout: float
    resid_dropout: float
    init_scale: float

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        tok, resets = x                     # (B, d_model), (B,)
        d, H = self.d_model, self.max_history

        buf = carry[..., :d]                # (B, H, d)  oldest .. newest
        valid = carry[..., d]               # (B, H)

        # 1. Episode boundary: everything before this step is unreachable.
        valid = jnp.where(resets[:, np.newaxis], 0.0, valid)

        # 2. Shift the window left by one and append the current token.
        buf = jnp.concatenate([buf[:, 1:], tok[:, np.newaxis, :]], axis=1)
        valid = jnp.concatenate([valid[:, 1:], jnp.ones_like(valid[:, :1])], axis=1)
        buf = buf * valid[..., np.newaxis]  # keep masked slots exactly zero

        # 3. Relative position code (slot i == H-1-i steps ago).
        h = buf
        if self.pos_encoding == "learned":
            pos = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (H, d))
            h = h + pos[np.newaxis]
        elif self.pos_encoding == "sinusoidal":
            h = h + _sinusoidal(H, d)[np.newaxis]

        # 4. Causal mask, intersected with slot validity. The diagonal stays
        #    open so no query row is fully masked (all -inf -> NaN softmax).
        causal = jnp.tril(jnp.ones((H, H), dtype=bool))[np.newaxis, np.newaxis]
        key_ok = valid.astype(bool)[:, np.newaxis, np.newaxis, :]
        mask = (causal & key_ok) | jnp.eye(H, dtype=bool)[np.newaxis, np.newaxis]

        # 5. Blocks.
        for _ in range(self.n_layers):
            h = TransformerBlock(
                d_model=d,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                activation=self.activation,
                norm_placement=self.norm_placement,
                qkv_bias=self.qkv_bias,
                attn_dropout=self.attn_dropout,
                resid_dropout=self.resid_dropout,
            )(h, mask, deterministic=True)

        # 6. Read out the CURRENT step only (slot H-1) and widen to out_dim so
        #    the downstream heads see the same width the GRU gave them.
        y = nn.LayerNorm()(h[:, -1, :])
        y = nn.Dense(
            self.out_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(y)

        new_carry = jnp.concatenate([buf, valid[..., np.newaxis]], axis=-1)
        return new_carry, y


class ScannedTransformer(nn.Module):
    """Drop-in replacement for `ScannedRNN`.

    Call: `carry, y = ScannedTransformer(**hp)(carry, (ins, resets))`
      ins    (T, B, d_in)  resets (T, B)
      carry  (B, max_history, d_model + 1)
      y      (T, B, out_dim)
    """
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    max_history: int
    out_dim: int
    activation: str = "gelu"
    norm_placement: str = "pre"
    pos_encoding: str = "learned"
    qkv_bias: bool = True
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        # Token projection is applied to the whole (T, B, .) tensor at once --
        # outside the per-step scan -- so the window only ever stores d_model.
        tok = nn.Dense(
            self.d_model,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name="token_proj",
        )(ins)
        return _ScannedTransformerCore(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            max_history=self.max_history,
            out_dim=self.out_dim,
            activation=self.activation,
            norm_placement=self.norm_placement,
            pos_encoding=self.pos_encoding,
            qkv_bias=self.qkv_bias,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            init_scale=self.init_scale,
            name="core",
        )(carry, (tok, resets))

    @staticmethod
    def initialize_carry(batch_size: int, max_history: int, d_model: int):
        """Empty window: all-zero tokens, every slot marked invalid."""
        return jnp.zeros((batch_size, max_history, d_model + 1), dtype=jnp.float32)

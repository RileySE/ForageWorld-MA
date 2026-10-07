"""
GTrXL (Gated Transformer-XL) sequence model for the PPO baselines.

A faithful port of the GTrXL baseline in the authors' AGaLiTe release

    Pramanik, Elelimy, Machado & White,
    "AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement
    Learning", TMLR 10/2024.  https://arxiv.org/abs/2310.15719
    code: https://github.com/subho406/agalite  (src_pure/models/gtrxl.py)

which is itself a JAX port of the DI-engine GTrXL (Parisotto et al., 2020),
the implementation the paper says it used (Appendix G.1).  `ScannedGTrXL` is a
**drop-in replacement for `ScannedRNN`**, same call contract as `ScannedRNN` /
`ScannedTransformer` / `ScannedAGaLiTe`:

    carry, y = ScannedGTrXL(...)(carry, (ins, resets))

        ins    : (T, B, d_in)   per-step embedding, the same tensor the GRU got
        resets : (T, B)         `last_done`; True => this step starts a new episode
        carry  : (B, carry_size(hp))   one flat float array, see "Carry layout"
        y      : (T, B, embedding_dim) (or GX_OUT_DIM if a projection is asked for)

Defaults are the paper's "GTrXL-128" Craftax configuration -- Table 4, GTrXL
column, with memory size M = 128 -- i.e. the authors' config_pure/craftax/
gtrxl128.yaml: d = 128, d_h = 64, 4 heads, 4 layers, mlp_num = 2, M = 128.

WHAT MAKES IT GTrXL RATHER THAN A PLAIN WINDOWED TRANSFORMER
------------------------------------------------------------
`transformer_policy.ScannedTransformer` caches raw token embeddings and
re-attends over them.  GTrXL is Transformer-XL plus RL-specific stabilisation:

  * Segment-level recurrence.  The carry holds, PER LAYER, the last M hidden
    states of that layer (memory), not just the input tokens.  Layer l attends
    over [memory_l ; current segment], so the receptive field compounds with
    depth -- nominally up to L*M steps, far beyond the M-step window.
  * Relative positional encoding (Dai et al., 2019).  Attention is
    (q + u)K^T + rel_shift((q + v)R^T), with sinusoidal R and two learned
    bias vectors u, v shared across layers.
  * Identity map reordering (pre-LN) + ReLU on the sublayer output.
  * GRU gating on both residual paths, with the update-gate bias initialised
    positive (GX_GRU_BIAS, default 2.0) so each block starts near the identity.
  * The memory is `stop_gradient`-ed, so no gradient flows across segments.

QUIRKS OF THE REFERENCE THAT ARE REPRODUCED HERE ON PURPOSE
------------------------------------------------------------
These look like mistakes.  They are what the published runs used, so we keep
them rather than silently "fixing" the baseline we are comparing against.

1.  The query is computed from the RAW layer input, while keys and values come
    from LayerNorm([memory ; input]).  Pre-LN normally normalises all three.
    See `GatedTransformerXLLayer.__call__` in the reference and DI-engine's
    `gtrxl.py`, which does the same.
2.  The MLP applies its activation after the LAST Dense too, so the FFN output
    is already ReLU'd before the gate's own ReLU-free path.
3.  Dense layers use flax's DEFAULT init (lecun_normal, zero bias), NOT the
    orthogonal init used elsewhere in this repo and in the authors' own
    agalite.py.  u and v are zero-initialised.

MASKING AND EPISODE BOUNDARIES
------------------------------
The carry keeps a per-slot boolean mask alongside the memory (True = blocked).
It starts all-True, so a fresh agent attends to nothing but itself.  At a step
where `resets` is True the mask is set all-True again, which severs attention
to everything before the boundary; the current position is then unmasked.  The
mask slides with the memory, so a slot stays visible until it falls out of the
window or a reset wipes it.

Carry layout
------------
One flat float array of shape (B, carry_size(hp)) so it stays a plain tensor
like the GRU hidden state (no pytree), which is what the existing minibatch /
checkpoint / hidden-state-logging code expects:

    [0 : (L+1)*M*d]   per-layer memory, reshaped (L+1, M, d).  Slot l is the
                      INPUT to layer l (slot 0 = the embedded token), which is
                      what Transformer-XL caches.
    [(L+1)*M*d : ]    the (M,) slot mask, stored as 1.0 = blocked.

Recompute semantics -- READ THIS BEFORE COMPARING PPO RATIOS
-------------------------------------------------------------
GTrXL's rollout and its PPO loss recomputation are NOT the same computation,
and the gap is architectural, not numerical.  During the rollout the model is
called one step at a time, so step t attends over exactly [t-M, t].  During the
recompute it is called once on a T-step segment, so step j of that segment
attends over [segment_start - M, segment_start + j] -- up to M + j steps back,
and the in-segment keys come from the current forward pass rather than from
memory.  The discrepancy compounds with depth.

There is exactly one regime where the two agree: a fully masked memory (a
fresh episode) and a segment no longer than M -- then both see precisely the
segment prefix.  Once the memory holds unmasked history, they diverge from the
segment's second step onward.

Measured, so the claim is not hand-waving.  On this repo's predators/passives
shape (M=128, T=64, memory populated), the sequence-model output differs by up
to 3.5e-2 between the two paths.  That is large, but the actor head compresses
it: the resulting epoch-0 importance ratio is 1 +/- 1.1e-4 (mean 1.3e-5), i.e.
0.055% of CLIP_EPS = 0.2, with no sample anywhere near the clip boundary.  For
scale, AGaLiTe's ratio error is 4.8e-7 and the GRU's is exactly 0 -- so GTrXL
is ~200x looser than AGaLiTe here, and still ~1800x inside the clip range.

The reference has exactly this property (its rollout is T=1 and its update is
T=NUM_STEPS), and so does every standard Transformer-XL RL setup, so we keep
it.  `transformer_policy` avoids the issue instead by recomputing its whole
window every step, at ~H times the FLOPs.  The two tests
`test_rollout_matches_recompute_when_the_memory_is_empty` and
`test_rollout_diverges_from_recompute_once_the_memory_is_populated` pin both
sides of this so the masking semantics cannot drift unnoticed.

Cost
----
`describe()` prints the parameter and carry budget.  With the GTrXL-128 config
and this repo's FC_DIM_SIZE = 128 / GRU_HIDDEN_DIM = 512, the core is ~1.6x the
GRU's parameters and the carry is (L+1)*M*d + M = 82,048 floats per env per
agent -- 160x the GRU's 512, and ~3x AGaLiTe's 26,625.  That carry is the
dominant memory cost of this policy; GX_MEMORY_LEN scales it linearly.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Defaults are the paper's GTrXL-128 Craftax configuration (Table 4, GTrXL
# column with M = 128), i.e. config_pure/craftax/gtrxl128.yaml upstream.
DEFAULT_HPARAMS: Dict[str, Any] = {
    "GX_LAYER_NUM":     4,      # L: GTrXL blocks
    "GX_EMBEDDING_DIM": 128,    # d: residual stream width
    "GX_HEAD_DIM":      64,     # d_h: per-head dim
    "GX_HEAD_NUM":      4,      # attention heads
    "GX_MLP_NUM":       2,      # Dense layers in each block's FFN
    "GX_MEMORY_LEN":    128,    # M: cached hidden states per layer  <-- the "128"
    "GX_GRU_BIAS":      2.0,    # b_g init for the two GRU gates per block
    "GX_RESET_ON_TERMINATE": True,   # sever attention across episode boundaries
    "GX_OUT_DIM":       None,   # None -> emit d, exactly like the reference
}


def hparams_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull GX_* keys out of a run config, filling in DEFAULT_HPARAMS."""
    hp = {k: config.get(k, v) for k, v in DEFAULT_HPARAMS.items()}
    for key in ("GX_LAYER_NUM", "GX_EMBEDDING_DIM", "GX_HEAD_DIM", "GX_HEAD_NUM",
                "GX_MLP_NUM", "GX_MEMORY_LEN"):
        if int(hp[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {hp[key]!r}")
    if hp["GX_OUT_DIM"] is not None and int(hp["GX_OUT_DIM"]) < 1:
        raise ValueError(f'GX_OUT_DIM must be null or >= 1, got {hp["GX_OUT_DIM"]!r}')
    return {
        "layer_num": int(hp["GX_LAYER_NUM"]),
        "embedding_dim": int(hp["GX_EMBEDDING_DIM"]),
        "head_dim": int(hp["GX_HEAD_DIM"]),
        "head_num": int(hp["GX_HEAD_NUM"]),
        "mlp_num": int(hp["GX_MLP_NUM"]),
        "memory_len": int(hp["GX_MEMORY_LEN"]),
        "gru_bias": float(hp["GX_GRU_BIAS"]),
        "reset_on_terminate": bool(hp["GX_RESET_ON_TERMINATE"]),
        "out_dim": None if hp["GX_OUT_DIM"] is None else int(hp["GX_OUT_DIM"]),
    }


# ---------------------------------------------------------------------------
# Carry layout
# ---------------------------------------------------------------------------
def memory_size(hp: Dict[str, Any]) -> int:
    """Floats of cached hidden state, for one env: (L+1) x M x d."""
    return (hp["layer_num"] + 1) * hp["memory_len"] * hp["embedding_dim"]


def carry_size(hp: Dict[str, Any]) -> int:
    """Total flat carry width: the memory plus the per-slot mask."""
    return memory_size(hp) + hp["memory_len"]


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------
def gru_core_params(input_dim: int, hidden: int) -> int:
    """Parameter count of the `ScannedRNN` core (flax GRUCell)."""
    return 3 * (input_dim * hidden + hidden) + 2 * (hidden * hidden) + (hidden * hidden + hidden)


def gtrxl_core_params(hp: Dict[str, Any], input_dim: int) -> int:
    """Parameter count of the `ScannedGTrXL` core.  Independent of memory_len."""
    d, dh, hn, L = hp["embedding_dim"], hp["head_dim"], hp["head_num"], hp["layer_num"]
    proj = hn * dh

    emb = input_dim * d + d
    uv = 2 * hn * dh                                  # shared content/position biases

    attn = (d * (2 * proj) + 2 * proj)                # fused K,V
    attn += d * proj + proj                           # Q
    attn += proj * d + d                              # output projection
    attn += d * proj + proj                           # positional projection
    gate = 6 * d * d + d
    # FFN: mlp_num Dense layers, dims [d] + [d]*(mlp_num-1) + [d] -- all d wide
    # because the reference passes embedding_dim as the hidden width.
    ffn = hp["mlp_num"] * (d * d + d)
    norms = 2 * (2 * d)

    total = emb + uv + L * (attn + 2 * gate + ffn + norms)
    if hp["out_dim"] is not None:
        total += d * hp["out_dim"] + hp["out_dim"]
    return total


def describe(hp: Dict[str, Any], input_dim: int, gru_hidden: int) -> str:
    """Human-readable hyperparameter + parameter/state-budget report."""
    gx_p = gtrxl_core_params(hp, input_dim)
    gru_p = gru_core_params(input_dim, gru_hidden)
    out_w = hp["out_dim"] if hp["out_dim"] is not None else hp["embedding_dim"]
    lines = [
        "-" * 74,
        f"GTrXL-{hp['memory_len']} policy - gated Transformer-XL sequence-model core",
        "  (Parisotto et al., 2020; port of github.com/subho406/agalite gtrxl.py)",
        "-" * 74,
        f"  GX_LAYER_NUM       {hp['layer_num']:<10}  GTrXL blocks (L)",
        f"  GX_EMBEDDING_DIM   {hp['embedding_dim']:<10}  residual stream width (d)",
        f"  GX_HEAD_DIM        {hp['head_dim']:<10}  d_h",
        f"  GX_HEAD_NUM        {hp['head_num']:<10}  attention heads",
        f"  GX_MLP_NUM         {hp['mlp_num']:<10}  Dense layers per FFN",
        f"  GX_MEMORY_LEN      {hp['memory_len']:<10}  M: cached hidden states per layer",
        f"  GX_GRU_BIAS        {hp['gru_bias']:<10}  b_g init for the GRU gates",
        f"  GX_RESET_ON_TERM.  {str(hp['reset_on_terminate']):<10}  sever attention at episode boundaries",
        f"  GX_OUT_DIM         {str(hp['out_dim']):<10}  output width = {out_w}",
        "-" * 74,
        f"  input dim (FC_DIM_SIZE)          {input_dim}",
        f"  gtrxl core params                {gx_p:,}",
        f"  GRU core params (h={gru_hidden})           {gru_p:,}",
        f"  ratio gtrxl / GRU                {gx_p / gru_p:.3f}x",
        f"  carry per env                    {carry_size(hp):,} floats   (GRU: {gru_hidden:,})",
        f"     memory (L+1) x M x d          {memory_size(hp):,}",
        f"     slot mask                     {hp['memory_len']:,}",
        f"  attention span                   {hp['memory_len']} steps/layer, "
        f"up to ~{hp['layer_num'] * hp['memory_len']} with depth",
        "-" * 74,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """Sinusoidal encoding of relative distance.  No learned parameters."""
    embedding_dim: int

    @nn.compact
    def __call__(self, pos_seq):
        inv_freq = 1.0 / (10000.0 ** (jnp.arange(0.0, self.embedding_dim, 2.0)
                                      / self.embedding_dim))
        sinusoid = jnp.outer(pos_seq, inv_freq)
        return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)


class GRUGatingUnit(nn.Module):
    """GTrXL gating layer (Parisotto et al., 2020, Eq. 8).

    Deliberately uses flax's DEFAULT Dense init, matching the reference's
    gtrxl.py -- note this differs from the orthogonal init in its agalite.py,
    so the two policies are not initialised the same way.
    """
    input_dim: int
    bg: float = 2.0

    def setup(self):
        dense = lambda: nn.Dense(self.input_dim, use_bias=False)
        self.Wr, self.Ur = dense(), dense()
        self.Wz, self.Uz = dense(), dense()
        self.Wg, self.Ug = dense(), dense()
        self.bgp = self.param("bgp", nn.initializers.constant(self.bg), (self.input_dim,))

    def __call__(self, x, y):
        r = nn.sigmoid(self.Wr(y) + self.Ur(x))
        z = nn.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = jnp.tanh(self.Wg(y) + self.Ug(r * x))
        return (1.0 - z) * x + z * h


class AttentionXL(nn.Module):
    """Transformer-XL relative multi-head attention (Dai et al., 2019, Eq. 6)."""
    input_dim: int
    head_num: int
    head_dim: int

    def setup(self):
        self.attention_kv = nn.Dense(self.head_num * self.head_dim * 2)
        self.attention_q = nn.Dense(self.head_num * self.head_dim)
        self.project = nn.Dense(self.input_dim)
        self.project_pos = nn.Dense(self.head_num * self.head_dim)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    @staticmethod
    def _rel_shift(x):
        """Shift row i of the relative-position scores left by i.

            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22
                                                a20 a21 a22

        Operates on the last two axes; x is (..., cur_seq, full_seq).
        """
        x_padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)])
        x_padded = x_padded.reshape(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        return x_padded[:, :, 1:].reshape(x.shape)

    def __call__(self, inputs, pos_embedding, full_input, u, v, mask):
        """inputs: (T,B,d) query source -- the RAW layer input, see quirk (1).
        full_input: (F,B,d) key/value source -- LayerNorm([memory ; input]).
        pos_embedding: (F,d).  u,v: (head_num, head_dim).  mask: (T,B,F), True = blocked.
        """
        T, B, d = inputs.shape
        F = full_input.shape[0]
        hn, dh = self.head_num, self.head_dim

        key, value = jnp.split(self.attention_kv(full_input), 2, axis=-1)
        key = key.reshape(F, B, hn, dh)
        value = value.reshape(F, B, hn, dh)
        query = self.attention_q(inputs).reshape(T, B, hn, dh)
        r = self.project_pos(pos_embedding).reshape(F, hn, dh)

        # (q + u) K^T  and  rel_shift((q + v) R^T)
        content = jnp.transpose(query + u, (1, 2, 0, 3)) @ jnp.transpose(key, (1, 2, 3, 0))
        position = jnp.transpose(query + v, (1, 2, 0, 3)) @ jnp.transpose(r, (1, 2, 0))
        attn = (content + self._rel_shift(position)) * self.scale   # (B,hn,T,F)

        # -1e20 rather than -inf: a fully masked row would otherwise be NaN.
        attn = jnp.where(jnp.transpose(mask, (1, 0, 2))[:, None], -1e20, attn)
        attn = nn.softmax(attn, axis=-1)

        out = attn @ jnp.transpose(value, (1, 2, 0, 3))             # (B,hn,T,dh)
        out = jnp.transpose(out, (2, 0, 1, 3)).reshape(T, B, hn * dh)
        return self.project(out)


class GatedTransformerXLLayer(nn.Module):
    """One GTrXL block: pre-LN attention + ReLU + GRU gate, then FFN + GRU gate."""
    input_dim: int
    head_dim: int
    hidden_dim: int
    head_num: int
    mlp_num: int
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, pos_embedding, u, v, memory, mask):
        full_input = jnp.concatenate([memory, inputs], axis=0)
        x1 = nn.LayerNorm(name="layernorm1")(full_input)
        # Quirk (1): query comes from `inputs`, not from the normalised x1.
        a1 = AttentionXL(self.input_dim, self.head_num, self.head_dim, name="attention")(
            inputs, pos_embedding, x1, u, v, mask
        )
        a1 = nn.relu(a1)
        o1 = GRUGatingUnit(self.input_dim, self.gru_bias, name="gate1")(inputs, a1)

        m2 = nn.LayerNorm(name="layernorm2")(o1)
        dims = [self.hidden_dim] * (self.mlp_num - 1) + [self.input_dim]
        for i, width in enumerate(dims):
            # Quirk (2): the activation is applied after the last Dense too.
            m2 = nn.relu(nn.Dense(width, name=f"mlp_{i}")(m2))
        return GRUGatingUnit(self.input_dim, self.gru_bias, name="gate2")(o1, m2)


class ScannedGTrXL(nn.Module):
    """Drop-in replacement for `ScannedRNN` / `ScannedTransformer` / `ScannedAGaLiTe`.

    Call: `carry, y = ScannedGTrXL(**hp)(carry, (ins, resets))`
      ins    (T, B, d_in)   resets (T, B)
      carry  (B, carry_size(hp))
      y      (T, B, embedding_dim)  -- or (T, B, out_dim) when `out_dim` is set
    """
    layer_num: int
    embedding_dim: int
    head_dim: int
    head_num: int
    mlp_num: int
    memory_len: int
    gru_bias: float = 2.0
    reset_on_terminate: bool = True
    out_dim: Optional[int] = None

    @property
    def _hp(self) -> Dict[str, Any]:
        return {"layer_num": self.layer_num, "memory_len": self.memory_len,
                "embedding_dim": self.embedding_dim}

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        T, B = ins.shape[0], ins.shape[1]
        L, M, d = self.layer_num, self.memory_len, self.embedding_dim
        F = M + T

        # -- unpack the flat carry -------------------------------------------
        mem = carry[:, :memory_size(self._hp)].reshape(B, L + 1, M, d)
        mem = jnp.transpose(mem, (1, 2, 0, 3))                  # (L+1, M, B, d)
        last_mask = carry[:, memory_size(self._hp):] > 0.5      # (B, M) True = blocked

        h = nn.relu(nn.Dense(d, name="embedding")(ins))

        # -- per-step attention mask ------------------------------------------
        # True = blocked.  A reset re-blocks everything; the current slot is then
        # opened, and stays open for later steps of the segment.
        def _step_mask(prev, xs):
            term, idx = xs
            if self.reset_on_terminate:
                prev = jnp.where(term[:, None] > 0.5, True, prev)
            cur = jnp.where(jnp.arange(F) == M + idx, False, prev)
            return cur, cur

        init_mask = jnp.concatenate([last_mask, jnp.ones((B, T), dtype=bool)], axis=-1)
        new_mask, attn_mask = jax.lax.scan(
            _step_mask, init_mask, (resets, jnp.arange(T))
        )                                                        # (B,F), (T,B,F)
        new_mask = new_mask[:, -M:]

        # -- relative positions, farthest first --------------------------------
        pos = jnp.arange(F - 1, -1, -1.0, dtype=ins.dtype)
        pos_embedding = PositionalEmbedding(d, name="pos_embedding")(pos)   # (F, d)

        u = self.param("u", nn.initializers.zeros, (self.head_num, self.head_dim))
        v = self.param("v", nn.initializers.zeros, (self.head_num, self.head_dim))

        hidden_states = [h]
        for i in range(L):
            h = GatedTransformerXLLayer(
                input_dim=d, head_dim=self.head_dim, hidden_dim=d,
                head_num=self.head_num, mlp_num=self.mlp_num,
                gru_bias=self.gru_bias, name=f"layer_{i}",
            )(h, pos_embedding, u, v, mem[i], attn_mask)
            hidden_states.append(h)

        # -- slide the memory window, detaching it from the graph --------------
        new_mem = jnp.stack(
            [jax.lax.stop_gradient(
                jnp.concatenate([mem[i], hidden_states[i]], axis=0)[T:F]
             ) for i in range(L + 1)],
            axis=0,
        )                                                        # (L+1, M, B, d)

        if self.out_dim is not None:
            h = nn.Dense(self.out_dim, name="out_proj")(h)

        new_carry = jnp.concatenate(
            [jnp.transpose(new_mem, (2, 0, 1, 3)).reshape(B, -1),
             new_mask.astype(carry.dtype)],
            axis=-1,
        )
        return new_carry, h

    @staticmethod
    def initialize_carry(batch_size: int, state_dim: int, memory_len: int):
        """Zero memory, and every slot masked so a fresh agent attends to itself only."""
        carry = jnp.zeros((batch_size, state_dim), dtype=jnp.float32)
        return carry.at[:, state_dim - memory_len:].set(1.0)

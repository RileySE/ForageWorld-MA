"""
AGaLiTe recurrent-attention sequence model for the PPO baselines.

A faithful port of the authors' released implementation of

    Pramanik, Elelimy, Machado & White,
    "AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement
    Learning", TMLR 10/2024.  https://arxiv.org/abs/2310.15719
    code: https://github.com/subho406/agalite  (src_pure/models/agalite.py)

`ScannedAGaLiTe` is a **drop-in replacement for `ScannedRNN`** in
seperate_ippo_rnn.py / central_ippo_rnn.py / ippo_rnn.py.  It keeps exactly the
same call contract as `ScannedRNN` and `ScannedTransformer`, so the surrounding
PPO machinery (rollout scan, GAE, minibatching, checkpointing) is unchanged:

    carry, y = ScannedAGaLiTe(...)(carry, (ins, resets))

        ins    : (T, B, d_in)   per-step embedding, the same tensor the GRU got
        resets : (T, B)         `last_done`; True => this step starts a new episode
        carry  : (B, carry_size(hp))   one flat float array, see "Carry layout"
        y      : (T, B, d_model)       (or AG_OUT_DIM if a projection is asked for)

WHY THIS INSTEAD OF A WINDOWED TRANSFORMER
------------------------------------------
`transformer_policy.ScannedTransformer` keeps the last H token embeddings and
re-attends over them every step: memory O(H*d), compute O(H^2*d) per step, and
it is *blind* to anything older than H steps.  AGaLiTe replaces softmax
attention with a gated linear-attention recurrence, so the agent carries a
fixed-size summary of the **entire** episode at O(r*eta*d_h) space and
O(d^2 + r*eta*d_h) time per step -- no context window, no horizon cut-off.
Craftax episodes run to 9000 steps, which is exactly the regime the paper
targets (Section 5, "Craftax", and Appendix G.5).

THE MATH (paper equation / algorithm numbers in brackets)
---------------------------------------------------------
Per head, with head dim `dh`, feature-map factor `eta` (so the key/query feature
dimension is dk = eta*dh), and f() the row-major flatten of a matrix:

    k_t = f(relu(W_p1 x_t) (x) relu(W_K x_t))            [Alg 4 line 2]
    q_t = f(relu(W_p2 x_t) (x) relu(W_Q x_t))            [Alg 4 line 3]
    v_t = W_V x_t                                        [Alg 4 line 4]
    b_t = sigmoid(W_beta x_t)                            [Alg 4 line 5]
    g_t = f(sigmoid(W_p3 x_t) (x) sigmoid(W_gamma x_t))  [Alg 4 line 6]

GaLiTe (Algorithm 3) keeps a full matrix state C_t in R^{dh x eta*dh}.  AGaLiTe
(Algorithm 4) approximates C_t by a sum of `r` outer products, derived from a
cosine approximation of the Kronecker delta [Eq. 20], so the state becomes a set
of *vectors* instead of a matrix -- one factor of dh cheaper:

    vt^i_t = cos(w_i t) (b_t . v_t) + (1 - b_t) . vt^i_{t-1}     [34]
    kt^i_t = cos(w_i t) (g_t . k_t) + (1 - g_t) . kt^i_{t-1}     [35]
    s_t    = g_t . k_t             + (1 - g_t) . s_{t-1}         [Alg 4 line 12]
    a_t    = sum_i vt^i_t (kt^i_t . q_t) / (2 r (s_t . q_t))     [36]

CONVENTIONS TAKEN FROM THE AUTHORS' CODE (not from the paper text)
------------------------------------------------------------------
The paper and the released code differ in two places.  We follow the **code**,
because that is what produced the published results; both differences are noted
here so the discrepancy is not silently inherited.

1.  `r` counts the cosine terms.  Algorithm 4 runs i = 0..r inclusive, i.e.
    r + 1 vectors; the code allocates exactly `r` vectors.  The released Craftax
    config (config_pure/craftax/arelit.yaml) sets `R: 2`, and Table 4 of the
    paper reports r = 1 for the same run -- so the code's `r` is the paper's
    r + 1.  `AG_R` here is the code's `r` (= number of cosine terms).
    ==> the paper's Craftax r = 1  is  AG_R: 2.

2.  The frequencies are `w = linspace(-pi, pi, r)`, not the paper's
    w_i = 2*pi*i/r.  With AG_R: 2 that gives w = (-pi, +pi), i.e.
    cos(w t) = (-1)^t for both terms.

`t` is a **global** step counter (`tick`).  The reference code deliberately does
*not* reset it on an episode boundary -- only the recurrent vectors are reset --
so we do the same.  It starts at 1.0 and step j of the run sees t = j + 2.

Numerical note (AG_STABLE_PHASE): the reference evaluates `cos(t * w)` directly.
Over a 2e9-step run `t` reaches ~4e6 per env, and `t * pi` then has an absolute
float32 error of order 1 radian, which turns the intended (-1)^t oscillation
into noise.  By default we instead evaluate `cos(2*pi*frac(t * w/(2*pi)))`,
which is *exactly* equal in real arithmetic and, for AG_R: 2, exact in float32
as well (w/(2*pi) = -/+0.5, and multiplying an integer by 0.5 is exact).  Set
AG_STABLE_PHASE: false to reproduce the reference bit-for-bit instead.

BLOCK STRUCTURE (GTrXL, Parisotto et al. 2020)
----------------------------------------------
Section 5: "we replace the XL-attention of GTrXL with one of the two approaches,
while preserving the order of the layers and the gating of GTrXL."  Each layer
is therefore identity-map-reordered (pre-LN), applies ReLU to the sublayer
output, and merges with a GRU-style gate whose update-gate bias is initialised
positive (AG_GRU_BIAS, default 2.0) so the block starts near the identity:

    e = relu(W_emb x)              (first layer only)
    a = attn(LayerNorm(e));   h = gate(e, relu(a))
    f = ffn (LayerNorm(h));   y = gate(h, relu(f))

Carry layout
------------
One flat float array of shape (B, carry_size(hp)) so it stays a plain tensor
like the GRU hidden state (no pytree), which is what the existing minibatch /
checkpoint / hidden-state-logging code expects.  Fields, in order:

    [0]                            global timestep counter `tick` (as a float)
    then, per layer l = 0..L-1:
      tilde_k  (r, n_heads, eta*dh)    key state    [35]
      tilde_v  (r, n_heads, dh)        value state  [34]
      s        (n_heads, eta*dh)       normaliser

Recompute semantics
-------------------
The rollout and the PPO loss recomputation run the identical code path, seeded
from the stored `init_hstate`, and the carry is a sufficient statistic for the
whole prefix (see `test_split_matches_whole`), so replaying a rollout reproduces
it.  The time recurrence is evaluated with `lax.associative_scan` (Appendix F),
so a length-T segment costs O(T log T) parallel work instead of T sequential
steps -- this is what the reference implementation does too.

One consequence is worth stating plainly, because it differs from the GRU:
`associative_scan`'s reduction tree depends on the segment length, so replaying
a T=1 rollout as one T=NUM_STEPS pass is *not* bit-identical, and the epoch-0
importance ratio is 1 +/- ~5e-7 rather than exactly 1.  Measured on this repo's
predators_20_moving config (4 agents x 64 steps x 64 envs): max |ratio - 1| =
4.8e-7, mean 8.3e-9, i.e. 2.4e-6 of CLIP_EPS = 0.2.  The GRU and the windowed
transformer give exactly 1 because their recurrences are strictly sequential.
Swapping `_discounted_sum` for a `lax.scan` would restore exactness at the cost
of T sequential steps per layer; the error is ~5 orders of magnitude below
anything PPO reacts to, so we keep the parallel scan.

Cost vs the GRU
---------------
`describe()` prints both the parameter count and the per-env carry size next to
the GRU core they replace.  With the paper's Craftax architecture (Table 4:
d = 128, dh = 64, 4 heads, 4 layers, eta = 8, r = 1) and this repo's
FC_DIM_SIZE = 128 / GRU_HIDDEN_DIM = 512, AGaLiTe is ~1.8x the GRU's parameters
and its carry is ~52x the GRU's 512 floats.  AG_ETA: 4 (the paper's other
Craftax-family value) halves the carry.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Every knob of the AGaLiTe policy, with its default.  Config keys are the
# upper-case names below.  Defaults are the paper's Craftax configuration
# (Table 4, AGaLiTe column + Section 5 "Craftax"), which is also what
# config_pure/craftax/arelit.yaml in the authors' repo runs.
DEFAULT_HPARAMS: Dict[str, Any] = {
    # -- width / depth -------------------------------------------------------
    "AG_N_LAYERS":      4,      # L: number of GTrXL-style blocks
    "AG_D_MODEL":       128,    # d: residual-stream width
    "AG_D_HEAD":        64,     # dh: per-head dim (independent of d_model)
    "AG_N_HEADS":       4,      # number of attention heads
    "AG_D_FFC":         128,    # FFN hidden width (the paper's config uses d)
    # -- recurrent attention -------------------------------------------------
    "AG_ETA":           8,      # eta: feature-map factor; dk = eta * dh
    "AG_R":             2,      # number of cosine terms == paper's r + 1.
                                #   Paper's Craftax r = 1  ->  AG_R: 2.
    # -- block internals -----------------------------------------------------
    "AG_GRU_BIAS":      2.0,    # b_g init for the GTrXL gate's update gate
    "AG_RESET_ON_TERMINATE": True,   # zero the recurrent state on `resets`
    # -- output --------------------------------------------------------------
    "AG_OUT_DIM":       None,   # None -> emit d_model, exactly like the
                                #   reference.  Set an int to add a final
                                #   Dense(out_dim) (e.g. GRU_HIDDEN_DIM) so the
                                #   actor/critic/aux heads see the GRU's width.
    # -- numerics ------------------------------------------------------------
    "AG_EPS":           1e-5,   # added to the 2*r*(s.q) denominator
    "AG_STABLE_PHASE":  True,   # see the "Numerical note" in the module docstring
}


def hparams_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull AG_* keys out of a run config, filling in DEFAULT_HPARAMS.

    Returns a plain dict of lower-case kwargs for `ScannedAGaLiTe`.
    """
    hp = {k: config.get(k, v) for k, v in DEFAULT_HPARAMS.items()}

    for key in ("AG_N_LAYERS", "AG_D_MODEL", "AG_D_HEAD", "AG_N_HEADS", "AG_D_FFC",
                "AG_ETA", "AG_R"):
        if int(hp[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {hp[key]!r}")
    if hp["AG_OUT_DIM"] is not None and int(hp["AG_OUT_DIM"]) < 1:
        raise ValueError(f'AG_OUT_DIM must be null or >= 1, got {hp["AG_OUT_DIM"]!r}')

    return {
        "n_layers": int(hp["AG_N_LAYERS"]),
        "d_model": int(hp["AG_D_MODEL"]),
        "d_head": int(hp["AG_D_HEAD"]),
        "n_heads": int(hp["AG_N_HEADS"]),
        "d_ffc": int(hp["AG_D_FFC"]),
        "eta": int(hp["AG_ETA"]),
        "r": int(hp["AG_R"]),
        "gru_bias": float(hp["AG_GRU_BIAS"]),
        "reset_on_terminate": bool(hp["AG_RESET_ON_TERMINATE"]),
        "out_dim": None if hp["AG_OUT_DIM"] is None else int(hp["AG_OUT_DIM"]),
        "eps": float(hp["AG_EPS"]),
        "stable_phase": bool(hp["AG_STABLE_PHASE"]),
    }


# ---------------------------------------------------------------------------
# Carry layout
# ---------------------------------------------------------------------------
def layer_state_size(hp: Dict[str, Any]) -> int:
    """Floats of recurrent state held by ONE layer, for one env."""
    h, dh, dk, r = hp["n_heads"], hp["d_head"], hp["eta"] * hp["d_head"], hp["r"]
    return r * h * dk + r * h * dh + h * dk       # tilde_k, tilde_v, s


def carry_size(hp: Dict[str, Any]) -> int:
    """Total flat carry width: one global tick + per-layer recurrent states."""
    return 1 + hp["n_layers"] * layer_state_size(hp)


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


def agalite_core_params(hp: Dict[str, Any], input_dim: int) -> int:
    """Parameter count of the `ScannedAGaLiTe` core.

    Independent of `r`: the approximation rank changes the *state*, not the
    parameters.
    """
    d, dh, h, eta = hp["d_model"], hp["d_head"], hp["n_heads"], hp["eta"]
    ffc, proj = hp["d_ffc"], h * dh

    emb = input_dim * d + d                       # first layer only
    attn = (d * (5 * proj) + 5 * proj)            # W_{K,Q,V,beta,gamma}, fused
    attn += d * (3 * h * eta) + 3 * h * eta       # W_{p1,p2,p3}, fused
    attn += proj * d + d                          # output projection
    gate = 6 * d * d + d                          # GTrXL GRU gate (bias-free W's + b_g)
    ffn = (d * ffc + ffc) + (ffc * d + d)
    norms = 2 * (2 * d)                           # two LayerNorms

    per_layer = attn + 2 * gate + ffn + norms
    total = emb + hp["n_layers"] * per_layer
    if hp["out_dim"] is not None:
        total += d * hp["out_dim"] + hp["out_dim"]
    return total


def describe(hp: Dict[str, Any], input_dim: int, gru_hidden: int) -> str:
    """Human-readable hyperparameter + parameter/state-budget report."""
    ag_p = agalite_core_params(hp, input_dim)
    gru_p = gru_core_params(input_dim, gru_hidden)
    dk = hp["eta"] * hp["d_head"]
    out_w = hp["out_dim"] if hp["out_dim"] is not None else hp["d_model"]
    lines = [
        "-" * 74,
        "AGaLiTe policy - gated linear recurrent-attention sequence-model core",
        "  (Pramanik et al., TMLR 2024; port of github.com/subho406/agalite)",
        "-" * 74,
        f"  AG_N_LAYERS        {hp['n_layers']:<10}  GTrXL-style blocks (L)",
        f"  AG_D_MODEL         {hp['d_model']:<10}  residual stream width (d)",
        f"  AG_D_HEAD          {hp['d_head']:<10}  dh (independent of d_model)",
        f"  AG_N_HEADS         {hp['n_heads']:<10}  attention heads",
        f"  AG_D_FFC           {hp['d_ffc']:<10}  FFN hidden width",
        f"  AG_ETA             {hp['eta']:<10}  feature map factor -> dk = {dk}",
        f"  AG_R               {hp['r']:<10}  cosine terms (== paper's r + 1, so paper r = {hp['r'] - 1})",
        f"  AG_GRU_BIAS        {hp['gru_bias']:<10}  b_g init for the GTrXL gates",
        f"  AG_RESET_ON_TERM.  {str(hp['reset_on_terminate']):<10}  zero recurrent state on episode boundary",
        f"  AG_OUT_DIM         {str(hp['out_dim']):<10}  output width = {out_w}",
        f"  AG_EPS             {hp['eps']:<10}  denominator floor",
        f"  AG_STABLE_PHASE    {str(hp['stable_phase']):<10}  float32-exact cos(w t) via phase folding",
        "-" * 74,
        f"  input dim (FC_DIM_SIZE)          {input_dim}",
        f"  agalite core params              {ag_p:,}",
        f"  GRU core params (h={gru_hidden})           {gru_p:,}",
        f"  ratio agalite / GRU              {ag_p / gru_p:.3f}x",
        f"  carry per env                    {carry_size(hp):,} floats   (GRU: {gru_hidden:,})",
        f"  context horizon                  unbounded (whole episode)",
        "-" * 74,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _binary_operator(x, y):
    """Associative combine for the first-order recurrence  h_t = a_t h_{t-1} + b_t."""
    a_i, b_i = x
    a_j, b_j = y
    return a_i * a_j, a_j * b_i + b_j


def _discounted_sum(start: jnp.ndarray, x: jnp.ndarray, discount: jnp.ndarray) -> jnp.ndarray:
    """h_t = discount_t * h_{t-1} + x_t for t = 0..T-1, with h_{-1} = `start`.

    Evaluated with a parallel prefix scan (Appendix F).  `discount` only has to
    broadcast against `x`; the leading axis of both is time.

        start    : (...)          x : (T, ...)      discount : (T, ...)
        returns  : (T, ...)
    """
    x_cat = jnp.concatenate([start[None], x], axis=0)
    ones = jnp.ones((1,) + discount.shape[1:], dtype=discount.dtype)
    d_cat = jnp.concatenate([ones, discount], axis=0)
    return jax.lax.associative_scan(_binary_operator, (d_cat, x_cat))[1][1:]


class GRUGatingUnit(nn.Module):
    """GTrXL gating layer (Parisotto et al., 2020, Eq. 8).

        r = sigmoid(Wr y + Ur x)
        z = sigmoid(Wz y + Uz x - b_g)
        h = tanh(Wg y + Ug (r . x))
        g(x, y) = (1 - z) . x + z . h

    `b_g` is initialised positive so z starts near 0 and the layer starts as the
    identity on x -- the property that makes GTrXL trainable in RL.  Weight
    names, shapes and inits mirror the reference implementation.
    """
    input_dim: int
    bg: float = 2.0

    def setup(self):
        dense = lambda: nn.Dense(self.input_dim, use_bias=False,
                                 kernel_init=orthogonal(np.sqrt(2)))
        self.Wr, self.Ur = dense(), dense()
        self.Wz, self.Uz = dense(), dense()
        self.Wg, self.Ug = dense(), dense()
        self.bgp = self.param("bgp", constant(self.bg), (self.input_dim,))

    def __call__(self, x, y):
        r = nn.sigmoid(self.Wr(y) + self.Ur(x))
        z = nn.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = jnp.tanh(self.Wg(y) + self.Ug(r * x))
        return (1.0 - z) * x + z * h


class AGaLiTeAttention(nn.Module):
    """AGaLiTe self-attention (Algorithm 4) over a whole (T, B, d) segment.

    Everything that does not depend on the previous state -- keys, queries,
    values, gating vectors -- is computed for all T timesteps at once with dense
    matmuls.  Only the first-order state recurrence is scanned over time, and it
    is done with `associative_scan` so the segment is processed in parallel
    (Appendix F).
    """
    input_dim: int          # width the attention output is projected back to
    d_head: int
    n_heads: int
    eta: int
    r: int
    eps: float = 1e-5
    reset_on_terminate: bool = True
    stable_phase: bool = True

    def setup(self):
        # Fused, exactly as in the reference: one Dense for K/Q/V/beta/gamma and
        # one for p1/p2/p3.  The fusion is not cosmetic -- orthogonal init over
        # the fused matrix is what the published runs used.
        self.Wkqvbg = nn.Dense(self.n_heads * self.d_head * 5,
                               kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.Wp = nn.Dense(self.n_heads * self.eta * 3,
                           kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.Wo = nn.Dense(self.input_dim,
                           kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

    def __call__(self, x, terminations, state, tick):
        """x: (T,B,d); terminations: (T,B) float; state: (B, layer_state_size);
        tick: (B,) global step counter.  Returns (out (T,B,d), new_state, new_tick)."""
        T, B = x.shape[0], x.shape[1]
        H, dh, e, r = self.n_heads, self.d_head, self.eta, self.r
        dk = e * dh

        # -- keys / queries / values / gating vectors, all timesteps at once ---
        kqvbg = self.Wkqvbg(x).reshape(T, B, H, 5 * dh)
        keys, queries, values, beta, gammas = jnp.split(kqvbg, 5, axis=-1)  # (T,B,H,dh)
        p1, p2, p3 = jnp.split(self.Wp(x).reshape(T, B, H, 3 * e), 3, axis=-1)  # (T,B,H,e)

        # f(a (x) b) with `a` the outer index: element (n, d) lands at n*dh + d.
        def outer(a, b):                                   # (T,B,H,e) x (T,B,H,dh)
            return (a[..., :, None] * b[..., None, :]).reshape(T, B, H, e * dh)

        keys = outer(nn.relu(p1), nn.relu(keys))           # k_t     [Alg 4 line 2]
        queries = outer(nn.relu(p2), nn.relu(queries))     # q_t     [Alg 4 line 3]
        gammas = outer(nn.sigmoid(p3), nn.sigmoid(gammas))  # gamma_t [Alg 4 line 6]
        beta = nn.sigmoid(beta)                            # beta_t  [Alg 4 line 5]

        # -- cosine coefficients cos(w_i t) -----------------------------------
        # `tick` is global and never reset (reference behaviour); step j of the
        # segment is t = tick + j + 1.
        ticks = tick[None, :, None] + jnp.arange(1, T + 1, dtype=x.dtype)[:, None, None]
        if self.stable_phase:
            # cos(2 pi frac(t * w/(2 pi))) == cos(t w), but keeps the argument in
            # [0, 2 pi) so float32 stays accurate for t up to 2**24.
            w_over_2pi = jnp.asarray(np.linspace(-0.5, 0.5, r), dtype=x.dtype)
            occil = jnp.cos(2.0 * np.pi * jnp.mod(ticks * w_over_2pi, 1.0))
        else:
            omegas = jnp.asarray(np.linspace(-np.pi, np.pi, r), dtype=x.dtype)
            occil = jnp.cos(ticks * omegas)                # (T,B,r)
        occil = occil[..., None, None]                     # (T,B,r,1,1)

        # -- state inputs and per-step decays ---------------------------------
        values = (values * beta)[:, :, None]               # (T,B,1,H,dh)
        values = values * occil                            # cos(w_i t)(beta.v)  [34]
        keys = keys * gammas                               # gamma.k
        s_in = keys                                        # normaliser input
        keys = keys[:, :, None] * occil                    # cos(w_i t)(gamma.k) [35]

        if self.reset_on_terminate:
            keep = (1.0 - terminations)[..., None, None]   # (T,B,1,1)
            d_gamma = (1.0 - gammas) * keep                # (T,B,H,dk)
            d_beta = (1.0 - beta) * keep                   # (T,B,H,dh)
        else:
            d_gamma, d_beta = 1.0 - gammas, 1.0 - beta

        # -- recurrent update, parallel over the time axis ---------------------
        tk0, tv0, s0 = self._unpack(state, B)
        tilde_k = _discounted_sum(tk0, keys, d_gamma[:, :, None])    # (T,B,r,H,dk)
        tilde_v = _discounted_sum(tv0, values, d_beta[:, :, None])   # (T,B,r,H,dh)
        s = _discounted_sum(s0, s_in, d_gamma)                       # (T,B,H,dk)

        # -- attention output, without ever forming C_t ------------------------  [36]
        kq = jnp.einsum("tbrhd,tbhd->tbrh", tilde_k, queries)
        kv = (tilde_v * kq[..., None]).sum(axis=2)                   # (T,B,H,dh)
        norm = jnp.einsum("tbhd,tbhd->tbh", s, queries)              # (T,B,H)
        attn = kv / (2.0 * r * norm[..., None] + self.eps)

        out = self.Wo(attn.reshape(T, B, H * dh))
        new_state = self._pack(tilde_k[-1], tilde_v[-1], s[-1])
        return out, new_state, tick + T

    # -- flat carry <-> per-field tensors -------------------------------------
    def _unpack(self, state, B):
        H, dh, dk, r = self.n_heads, self.d_head, self.eta * self.d_head, self.r
        shapes = [(r, H, dk), (r, H, dh), (H, dk)]
        out, off = [], 0
        for shp in shapes:
            n = int(np.prod(shp))
            out.append(state[:, off:off + n].reshape((B,) + shp))
            off += n
        assert off == state.shape[1], f"carry width {state.shape[1]} != expected {off}"
        return tuple(out)

    @staticmethod
    def _pack(*fields):
        return jnp.concatenate([f.reshape(f.shape[0], -1) for f in fields], axis=-1)


class AGaLiTeBlock(nn.Module):
    """One GTrXL block with AGaLiTe recurrent attention in place of XL-attention.

    `use_dense` mirrors the reference: only the first layer carries the input
    embedding Dense (+ ReLU); later layers consume the previous layer's output.
    """
    d_model: int
    d_head: int
    d_ffc: int
    n_heads: int
    eta: int
    r: int
    use_dense: bool = False
    gru_bias: float = 2.0
    eps: float = 1e-5
    reset_on_terminate: bool = True
    stable_phase: bool = True

    @nn.compact
    def __call__(self, x, terminations, state, tick):
        if self.use_dense:
            x = nn.Dense(self.d_model, name="emb_layer",
                         kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)

        # Identity map reordering (pre-LN) + ReLU on the sublayer + GRU gate.
        attn_out, new_state, new_tick = AGaLiTeAttention(
            input_dim=self.d_model, d_head=self.d_head, n_heads=self.n_heads,
            eta=self.eta, r=self.r, eps=self.eps,
            reset_on_terminate=self.reset_on_terminate,
            stable_phase=self.stable_phase, name="attn",
        )(nn.LayerNorm(name="ln_attn")(x), terminations, state, tick)
        h = GRUGatingUnit(self.d_model, self.gru_bias, name="gate_attn")(x, nn.relu(attn_out))

        f = nn.LayerNorm(name="ln_ffn")(h)
        f = nn.Dense(self.d_ffc, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0), name="ffn_1")(f)
        f = nn.relu(f)
        f = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0), name="ffn_2")(f)
        out = GRUGatingUnit(self.d_model, self.gru_bias, name="gate_ffn")(h, nn.relu(f))
        return out, new_state, new_tick


class ScannedAGaLiTe(nn.Module):
    """Drop-in replacement for `ScannedRNN` / `ScannedTransformer`.

    Call: `carry, y = ScannedAGaLiTe(**hp)(carry, (ins, resets))`
      ins    (T, B, d_in)   resets (T, B)
      carry  (B, carry_size(hp))
      y      (T, B, d_model)  -- or (T, B, out_dim) when `out_dim` is set
    """
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_ffc: int
    eta: int
    r: int
    gru_bias: float = 2.0
    reset_on_terminate: bool = True
    out_dim: Optional[int] = None
    eps: float = 1e-5
    stable_phase: bool = True

    @property
    def _hp(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers, "n_heads": self.n_heads, "d_head": self.d_head,
            "eta": self.eta, "r": self.r,
        }

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        terminations = resets.astype(ins.dtype)

        per_layer = layer_state_size(self._hp)
        tick = carry[:, 0]
        states = [carry[:, 1 + i * per_layer: 1 + (i + 1) * per_layer]
                  for i in range(self.n_layers)]

        h, new_states, new_tick = ins, [], tick
        for i in range(self.n_layers):
            # Every layer advances the same clock by T, so they stay in lockstep;
            # we keep one copy of it in the carry.
            h, s_new, new_tick = AGaLiTeBlock(
                d_model=self.d_model, d_head=self.d_head, d_ffc=self.d_ffc,
                n_heads=self.n_heads, eta=self.eta, r=self.r,
                use_dense=(i == 0), gru_bias=self.gru_bias, eps=self.eps,
                reset_on_terminate=self.reset_on_terminate,
                stable_phase=self.stable_phase, name=f"block_{i + 1}",
            )(h, terminations, states[i], tick)
            new_states.append(s_new)

        if self.out_dim is not None:
            h = nn.Dense(self.out_dim, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0), name="out_proj")(h)

        new_carry = jnp.concatenate([new_tick[:, None]] + new_states, axis=-1)
        return new_carry, h

    @staticmethod
    def initialize_carry(batch_size: int, state_dim: int):
        """Zero recurrent state; the global tick starts at 1.0 (reference value)."""
        carry = jnp.zeros((batch_size, state_dim), dtype=jnp.float32)
        return carry.at[:, 0].set(1.0)

"""Tests for baselines/agalite_policy.py (the AGaLiTe sequence model).

Two groups:

* Always-on: the invariants the PPO machinery in seperate_ippo_rnn.py relies on
  -- carry width, split-vs-whole equivalence (the PPO recompute crosses rollout
  boundaries), bit-exact recompute (importance ratio == 1 at epoch 0), episode
  resets actually cutting history, and no cross-env leakage.

* Opt-in: numerical equivalence with the authors' released implementation.
  Clone https://github.com/subho406/agalite and point AGALITE_REF_DIR at it:

      git clone https://github.com/subho406/agalite /tmp/agalite
      AGALITE_REF_DIR=/tmp/agalite pytest baselines/tests/test_agalite_policy.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agalite_policy as agp  # noqa: E402


SMALL = dict(n_layers=2, d_model=16, d_head=8, d_ffc=16, n_heads=2, eta=3, r=2)
T, B = 12, 6


def _build(**overrides):
    hp = {**SMALL, **overrides}
    model = agp.ScannedAGaLiTe(**hp)
    carry = agp.ScannedAGaLiTe.initialize_carry(B, agp.carry_size(hp))
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    ins = jax.random.normal(k1, (T, B, hp["d_model"]))
    resets = (jax.random.uniform(k2, (T, B)) < 0.3).astype(jnp.float32)
    params = {"params": model.init(k3, carry, (ins, resets))["params"]}
    return model, params, carry, ins, resets


def test_carry_size_and_init():
    hp = dict(SMALL)
    per_layer = hp["r"] * hp["n_heads"] * hp["eta"] * hp["d_head"] \
        + hp["r"] * hp["n_heads"] * hp["d_head"] + hp["n_heads"] * hp["eta"] * hp["d_head"]
    assert agp.layer_state_size(hp) == per_layer
    assert agp.carry_size(hp) == 1 + hp["n_layers"] * per_layer

    carry = agp.ScannedAGaLiTe.initialize_carry(B, agp.carry_size(hp))
    assert carry.shape == (B, agp.carry_size(hp))
    # tick starts at 1.0 (the reference's value); everything else is zero.
    np.testing.assert_array_equal(np.asarray(carry[:, 0]), np.ones(B))
    np.testing.assert_array_equal(np.asarray(carry[:, 1:]), np.zeros((B, agp.carry_size(hp) - 1)))


def test_shapes_and_tick_advances_by_T():
    model, params, carry, ins, resets = _build()
    new_carry, y = model.apply(params, carry, (ins, resets))
    assert y.shape == (T, B, SMALL["d_model"])
    assert new_carry.shape == carry.shape
    np.testing.assert_allclose(np.asarray(new_carry[:, 0]), np.asarray(carry[:, 0]) + T)


def test_out_dim_projection():
    model, params, carry, ins, resets = _build(out_dim=32)
    _, y = model.apply(params, carry, (ins, resets))
    assert y.shape == (T, B, 32)


def test_split_matches_whole():
    """Two consecutive segments must equal one long pass.

    This is what makes the PPO loss recomputation exact: the update replays a
    rollout starting from the stored carry, so the carry has to be a sufficient
    statistic for everything before it.
    """
    model, params, carry, ins, resets = _build()
    full_carry, full_y = model.apply(params, carry, (ins, resets))
    mid = 5
    c_a, y_a = model.apply(params, carry, (ins[:mid], resets[:mid]))
    c_b, y_b = model.apply(params, c_a, (ins[mid:], resets[mid:]))
    np.testing.assert_allclose(np.asarray(jnp.concatenate([y_a, y_b])),
                               np.asarray(full_y), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(c_b), np.asarray(full_carry), rtol=0, atol=1e-5)


def test_repeated_apply_is_bit_exact():
    """Same carry + same inputs + same segment length -> bit-identical output."""
    model, params, carry, ins, resets = _build()
    a = model.apply(params, carry, (ins, resets))[1]
    b = model.apply(params, carry, (ins, resets))[1]
    assert bool(jnp.all(a == b))


def test_rollout_vs_recompute_stays_within_float_noise():
    """Stepping T=1 at a time (the rollout) vs one T-long pass (the PPO
    recompute) is NOT bit-identical: `associative_scan`'s reduction tree depends
    on the segment length.  Pin the size of that gap -- it is what perturbs the
    epoch-0 importance ratio away from exactly 1, and it must stay ~5 orders of
    magnitude below CLIP_EPS.  See "Recompute semantics" in agalite_policy.py.
    """
    model, params, carry, ins, resets = _build()
    _, whole = model.apply(params, carry, (ins, resets))
    c, steps = carry, []
    for t in range(T):
        c, y_t = model.apply(params, c, (ins[t:t + 1], resets[t:t + 1]))
        steps.append(y_t)
    stepped = jnp.concatenate(steps, axis=0)
    gap = float(jnp.max(jnp.abs(stepped - whole)))
    scale = float(jnp.max(jnp.abs(whole)))
    assert 0.0 < gap < 1e-5 * max(scale, 1.0), (gap, scale)


def test_reset_cuts_history():
    model, params, carry, ins, resets = _build()
    r = np.zeros((T, B), np.float32)
    r[6, 0] = 1.0
    r = jnp.asarray(r)
    ins2 = ins.at[:6, 0].set(jax.random.normal(jax.random.PRNGKey(9), (6, SMALL["d_model"])))
    y1 = model.apply(params, carry, (ins, r))[1]
    y2 = model.apply(params, carry, (ins2, r))[1]
    # Everything from the reset onward is independent of the pre-reset inputs...
    np.testing.assert_array_equal(np.asarray(y1[6:, 0]), np.asarray(y2[6:, 0]))
    # ...and the control: before the reset the two DO differ, so the test above
    # is not passing vacuously.
    assert float(jnp.max(jnp.abs(y1[:6, 0] - y2[:6, 0]))) > 1e-3


def test_no_cross_env_leakage():
    model, params, carry, ins, resets = _build()
    r = np.zeros((T, B), np.float32)
    r[6, 0] = 1.0
    y_none = model.apply(params, carry, (ins, jnp.zeros((T, B), jnp.float32)))[1]
    y_one = model.apply(params, carry, (ins, jnp.asarray(r)))[1]
    np.testing.assert_array_equal(np.asarray(y_none[:, 1:]), np.asarray(y_one[:, 1:]))


def test_stable_phase_matches_reference_formula_at_small_t():
    """AG_STABLE_PHASE only folds the phase; it must not change the answer."""
    hp = dict(SMALL)
    stable = agp.ScannedAGaLiTe(**hp, stable_phase=True)
    plain = agp.ScannedAGaLiTe(**hp, stable_phase=False)
    _, params, carry, ins, resets = _build()
    a = stable.apply(params, carry, (ins, resets))[1]
    b = plain.apply(params, carry, (ins, resets))[1]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-5)


def test_hparams_from_config_defaults_to_paper_craftax():
    """Table 4 (AGaLiTe) + Section 5 "Craftax": d=128, dh=64, 4 heads, 4 layers,
    eta=8, r=1 -- and AG_R is the reference's convention, i.e. paper r + 1."""
    hp = agp.hparams_from_config({"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 512})
    assert (hp["n_layers"], hp["d_model"], hp["d_head"], hp["n_heads"]) == (4, 128, 64, 4)
    assert (hp["d_ffc"], hp["eta"], hp["r"]) == (128, 8, 2)
    assert hp["out_dim"] is None
    assert agp.carry_size(hp) == 26625
    assert agp.agalite_core_params(hp, 128) == 1779712


@pytest.mark.parametrize("key", ["AG_N_LAYERS", "AG_D_MODEL", "AG_ETA", "AG_R"])
def test_hparams_reject_nonpositive(key):
    with pytest.raises(ValueError):
        agp.hparams_from_config({"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 512, key: 0})


def test_gradients_are_finite():
    model, params, carry, ins, resets = _build()

    def loss(p):
        return jnp.sum(model.apply(p, carry, (ins, resets))[1] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


def test_vmap_over_agents():
    """seperate_ippo_rnn.py vmaps the whole network over a leading agent axis."""
    n_agents = 3
    hp = dict(SMALL)
    model = agp.ScannedAGaLiTe(**hp)
    cs = agp.carry_size(hp)
    rngs = jax.random.split(jax.random.PRNGKey(1), n_agents)
    carry = agp.ScannedAGaLiTe.initialize_carry(B, cs)
    ins = jnp.ones((T, B, hp["d_model"]))
    resets = jnp.zeros((T, B))
    params = jax.vmap(lambda k: model.init(k, carry, (ins, resets))["params"])(rngs)
    stacked_carry = jnp.tile(carry[None], (n_agents, 1, 1))
    stacked_ins = jnp.tile(ins[None], (n_agents, 1, 1, 1))
    stacked_resets = jnp.tile(resets[None], (n_agents, 1, 1))
    out_carry, y = jax.vmap(
        lambda p, h, i, d: model.apply({"params": p}, h, (i, d))
    )(params, stacked_carry, stacked_ins, stacked_resets)
    assert y.shape == (n_agents, T, B, hp["d_model"])
    assert out_carry.shape == (n_agents, B, cs)


# ---------------------------------------------------------------------------
# Equivalence with the authors' released implementation (opt-in).
# ---------------------------------------------------------------------------
_REF = os.environ.get("AGALITE_REF_DIR")


@pytest.mark.skipif(not _REF, reason="set AGALITE_REF_DIR to github.com/subho406/agalite")
def test_matches_reference_implementation():
    from flax.traverse_util import flatten_dict, unflatten_dict

    sys.path.insert(0, _REF)
    from src_pure.models.agalite import BatchedAGaLiTe  # noqa: E402

    L, D, DH, DFFC, HEADS, ETA, R = 3, 16, 8, 24, 2, 3, 4
    hp = dict(n_layers=L, d_model=D, d_head=DH, d_ffc=DFFC, n_heads=HEADS, eta=ETA, r=R)
    # stable_phase=False reproduces the reference's cos(t*w) bit-for-bit.
    mine = agp.ScannedAGaLiTe(**hp, stable_phase=False)
    ref = BatchedAGaLiTe(n_layers=L, d_model=D, d_head=DH, d_ffc=DFFC,
                         n_heads=HEADS, eta=ETA, r=R)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    ins = jax.random.normal(k1, (7, 5, D))
    resets = (jax.random.uniform(k2, (7, 5)) < 0.25).astype(jnp.float32)
    my_carry = agp.ScannedAGaLiTe.initialize_carry(5, agp.carry_size(hp))
    ref_carry = BatchedAGaLiTe.initialize_carry(5, L, HEADS, DH, ETA, R)
    my_params = mine.init(k3, my_carry, (ins, resets))["params"]

    gate = ["Wr", "Ur", "Wz", "Uz", "Wg", "Ug"]
    flat, out = flatten_dict(my_params), {}
    for i in range(1, L + 1):
        src, dst = f"block_{i}", f"layer{i}"
        pairs = [("ln_attn", "LayerNorm_0"), ("ln_ffn", "LayerNorm_1"),
                 ("ffn_1", "Dense_0"), ("ffn_2", "Dense_1")]
        if i == 1:
            pairs.append(("emb_layer", "emb_layer"))
        for a, b in pairs:
            for leaf in ("kernel", "bias", "scale"):
                if (src, a, leaf) in flat:
                    out[("VmapAGaLiTe_0", dst, b, leaf)] = flat[(src, a, leaf)]
        for a, b in [("Wkqvbg", "linear_kqvbetagammas"), ("Wp", "linear_p1p2p3"),
                     ("Wo", "project")]:
            for leaf in ("kernel", "bias"):
                out[("VmapAGaLiTe_0", dst, "AttentionORLiTLayer_0", b, leaf)] = \
                    flat[(src, "attn", a, leaf)]
        for a, b in [("gate_attn", "GRUGatingUnit_0"), ("gate_ffn", "GRUGatingUnit_1")]:
            out[("VmapAGaLiTe_0", dst, b, "bgp")] = flat[(src, a, "bgp")]
            for g in gate:
                out[("VmapAGaLiTe_0", dst, b, g, "kernel")] = flat[(src, a, g, "kernel")]
    ref_params = unflatten_dict(out)

    # Same parameters, same shapes, same names -- so this is a real comparison
    # and not two models that merely happen to have the same output width.
    want = jax.eval_shape(lambda: ref.init(k3, ref_carry, (ins, resets))["params"])
    assert {k: v.shape for k, v in flatten_dict(ref_params).items()} == \
           {k: v.shape for k, v in flatten_dict(want).items()}

    my_new, my_y = mine.apply({"params": my_params}, my_carry, (ins, resets))
    ref_new, ref_y = ref.apply({"params": ref_params}, ref_carry, (ins, resets))
    np.testing.assert_allclose(np.asarray(my_y), np.asarray(ref_y), rtol=0, atol=1e-4)

    per = agp.layer_state_size(hp)
    n1, n2 = R * HEADS * ETA * DH, R * HEADS * DH
    for i in range(1, L + 1):
        tk, tv, s, tick = ref_new[f"layer_{i}"]
        blk = my_new[:, 1 + (i - 1) * per: 1 + i * per]
        for got, want_ in [
            (blk[:, :n1].reshape(5, R, HEADS, ETA * DH), tk),
            (blk[:, n1:n1 + n2].reshape(5, R, HEADS, DH), tv),
            (blk[:, n1 + n2:].reshape(5, HEADS, ETA * DH), s),
        ]:
            np.testing.assert_allclose(np.asarray(got), np.asarray(want_), rtol=0, atol=1e-4)
        assert float(my_new[0, 0]) == float(tick[0, 0])

"""Tests for baselines/gtrxl_policy.py (the GTrXL sequence model).

Always-on tests cover the invariants seperate_ippo_rnn.py relies on, plus the
two GTrXL-specific behaviours that are easy to get wrong: the relative-position
shift and the episode-boundary masking.

The equivalence test against the authors' released implementation is opt-in:

    git clone https://github.com/subho406/agalite /tmp/agalite
    AGALITE_REF_DIR=/tmp/agalite pytest baselines/tests/test_gtrxl_policy.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gtrxl_policy as gxp  # noqa: E402


SMALL = dict(layer_num=2, embedding_dim=16, head_dim=8, head_num=2, mlp_num=2, memory_len=6)
T, B = 8, 4


def _build(**overrides):
    hp = {**SMALL, **overrides}
    model = gxp.ScannedGTrXL(**hp)
    carry = gxp.ScannedGTrXL.initialize_carry(B, gxp.carry_size(hp), hp["memory_len"])
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    ins = jax.random.normal(k1, (T, B, hp["embedding_dim"]))
    resets = (jax.random.uniform(k2, (T, B)) < 0.25).astype(jnp.float32)
    params = {"params": model.init(k3, carry, (ins, resets))["params"]}
    return model, params, carry, ins, resets


def test_carry_size_and_init():
    hp = dict(SMALL)
    L, M, d = hp["layer_num"], hp["memory_len"], hp["embedding_dim"]
    assert gxp.memory_size(hp) == (L + 1) * M * d
    assert gxp.carry_size(hp) == (L + 1) * M * d + M

    carry = gxp.ScannedGTrXL.initialize_carry(B, gxp.carry_size(hp), M)
    assert carry.shape == (B, gxp.carry_size(hp))
    # memory zeroed, every slot masked -> a fresh agent attends to itself only
    np.testing.assert_array_equal(np.asarray(carry[:, :gxp.memory_size(hp)]),
                                  np.zeros((B, gxp.memory_size(hp))))
    np.testing.assert_array_equal(np.asarray(carry[:, gxp.memory_size(hp):]),
                                  np.ones((B, M)))


def test_shapes():
    model, params, carry, ins, resets = _build()
    new_carry, y = model.apply(params, carry, (ins, resets))
    assert y.shape == (T, B, SMALL["embedding_dim"])
    assert new_carry.shape == carry.shape


def test_out_dim_projection():
    model, params, carry, ins, resets = _build(out_dim=32)
    _, y = model.apply(params, carry, (ins, resets))
    assert y.shape == (T, B, 32)


def test_memory_slides_and_holds_layer_inputs():
    """After a T-step segment the memory must be the last M layer inputs."""
    hp = dict(SMALL)
    L, M, d = hp["layer_num"], hp["memory_len"], hp["embedding_dim"]
    model, params, carry, ins, resets = _build()
    new_carry, _ = model.apply(params, carry, (ins, jnp.zeros((T, B))))
    mem = new_carry[:, :gxp.memory_size(hp)].reshape(B, L + 1, M, d)
    # T=8 > M=6, so the window is entirely new content: slot 0 of the memory is
    # the embedded token, which we can recompute independently.
    emb = jax.nn.relu(
        jnp.einsum("tbi,io->tbo", ins, params["params"]["embedding"]["kernel"])
        + params["params"]["embedding"]["bias"]
    )
    np.testing.assert_allclose(np.asarray(mem[:, 0]),
                               np.asarray(jnp.transpose(emb[T - M:], (1, 0, 2))),
                               rtol=0, atol=1e-5)


def test_repeated_apply_is_bit_exact():
    model, params, carry, ins, resets = _build()
    a = model.apply(params, carry, (ins, resets))[1]
    b = model.apply(params, carry, (ins, resets))[1]
    assert bool(jnp.all(a == b))


def test_reset_cuts_attention_to_the_past():
    """A reset must sever attention to everything before it."""
    model, params, carry, ins, resets = _build()
    r = np.zeros((T, B), np.float32)
    r[5, 0] = 1.0
    r = jnp.asarray(r)
    ins2 = ins.at[:5, 0].set(jax.random.normal(jax.random.PRNGKey(9), (5, SMALL["embedding_dim"])))
    y1 = model.apply(params, carry, (ins, r))[1]
    y2 = model.apply(params, carry, (ins2, r))[1]
    np.testing.assert_allclose(np.asarray(y1[5:, 0]), np.asarray(y2[5:, 0]),
                               rtol=0, atol=1e-6)
    # control: before the reset the two genuinely differ
    assert float(jnp.max(jnp.abs(y1[:5, 0] - y2[:5, 0]))) > 1e-3


def test_no_cross_env_leakage():
    model, params, carry, ins, resets = _build()
    r = np.zeros((T, B), np.float32)
    r[5, 0] = 1.0
    y_none = model.apply(params, carry, (ins, jnp.zeros((T, B))))[1]
    y_one = model.apply(params, carry, (ins, jnp.asarray(r)))[1]
    np.testing.assert_array_equal(np.asarray(y_none[:, 1:]), np.asarray(y_one[:, 1:]))


def test_fresh_carry_first_step_sees_only_itself():
    """Every memory slot starts masked, so step 0 of a fresh agent must not
    depend on the (zeroed) memory contents at all."""
    hp = dict(SMALL)
    model, params, carry, ins, _ = _build()
    poisoned = carry.at[:, :gxp.memory_size(hp)].set(7.0)   # garbage memory...
    zeros = jnp.zeros((1, B))
    y_clean = model.apply(params, carry, (ins[:1], zeros))[1]
    y_dirty = model.apply(params, poisoned, (ins[:1], zeros))[1]
    # ...must be ignored, because the mask blocks every slot.
    np.testing.assert_allclose(np.asarray(y_clean), np.asarray(y_dirty), rtol=0, atol=1e-6)


def test_rel_shift_matches_the_docstring_example():
    x = jnp.arange(9, dtype=jnp.float32).reshape(1, 1, 3, 3)   # a00..a22
    got = np.asarray(gxp.AttentionXL._rel_shift(x))[0, 0]
    # row i shifted left by i, pulling in from the row above (Dai et al. 2019)
    want = np.array([[2., 0., 3.], [4., 5., 0.], [6., 7., 8.]], dtype=np.float32)
    np.testing.assert_array_equal(got, want)


def test_positional_embedding_is_sinusoidal_and_paramless():
    d = 16
    m = gxp.PositionalEmbedding(d)
    p = m.init(jax.random.PRNGKey(0), jnp.arange(5.0))
    assert not jax.tree_util.tree_leaves(p)          # no learned parameters
    out = m.apply(p, jnp.arange(5.0))
    assert out.shape == (5, d)
    np.testing.assert_allclose(np.asarray(out[0]),
                               np.concatenate([np.zeros(d // 2), np.ones(d // 2)]),
                               rtol=0, atol=1e-6)    # pos 0 -> sin=0, cos=1


def test_rollout_matches_recompute_when_the_memory_is_empty():
    """The one regime where GTrXL's rollout and PPO recompute DO agree: a fresh
    carry (every memory slot masked) and a segment no longer than M.  Then both
    see exactly the segment prefix, so they are identical up to float noise.
    """
    hp = dict(SMALL)
    M = hp["memory_len"]
    t = M - 1
    model = gxp.ScannedGTrXL(**hp)
    carry = gxp.ScannedGTrXL.initialize_carry(B, gxp.carry_size(hp), M)
    ins = jax.random.normal(jax.random.PRNGKey(3), (t, B, hp["embedding_dim"]))
    zeros = jnp.zeros((t, B))
    params = {"params": model.init(jax.random.PRNGKey(0), carry, (ins, zeros))["params"]}

    _, whole = model.apply(params, carry, (ins, zeros))
    c, steps = carry, []
    for i in range(t):
        c, y = model.apply(params, c, (ins[i:i + 1], zeros[i:i + 1]))
        steps.append(y)
    np.testing.assert_allclose(np.asarray(jnp.concatenate(steps, 0)), np.asarray(whole),
                               rtol=0, atol=1e-5)


def test_rollout_diverges_from_recompute_once_the_memory_is_populated():
    """...and the regime where they DON'T, which is the one training runs in.

    Once the memory holds unmasked history, a T-step recompute lets step j
    attend M+j steps back while the rollout only ever saw M.  The gap is
    architectural (Transformer-XL segment recurrence), matches the reference,
    and is why the epoch-0 importance ratio is not ~1 for gtrxl as it is for
    gru/agalite.  Pin it so the masking semantics can't change unnoticed.
    """
    hp = dict(SMALL)
    M, t = hp["memory_len"], 4
    model = gxp.ScannedGTrXL(**hp)
    carry = gxp.ScannedGTrXL.initialize_carry(B, gxp.carry_size(hp), M)
    ins = jax.random.normal(jax.random.PRNGKey(4), (2 * M + t, B, hp["embedding_dim"]))
    zeros = jnp.zeros((2 * M + t, B))
    params = {"params": model.init(jax.random.PRNGKey(0), carry, (ins[:1], zeros[:1]))["params"]}

    # Warm the carry one step at a time, exactly as the rollout does, so the
    # memory is full and unmasked before the segment we compare.
    warm = 2 * M
    for i in range(warm):
        carry, _ = model.apply(params, carry, (ins[i:i + 1], zeros[i:i + 1]))

    _, whole = model.apply(params, carry, (ins[warm:], zeros[warm:]))
    c, steps = carry, []
    for i in range(warm, warm + t):
        c, y = model.apply(params, c, (ins[i:i + 1], zeros[i:i + 1]))
        steps.append(y)
    per_step = jnp.max(jnp.abs(jnp.concatenate(steps, 0) - whole), axis=(1, 2))

    # Step 0 of the segment still agrees (same memory, same span); step 1 on
    # does not, because the recompute can see one extra step of history.
    assert float(per_step[0]) < 1e-5
    assert float(per_step[1]) > 1e-4, (
        "expected GTrXL's rollout and recompute to diverge once the memory is "
        "populated; if they now agree, the memory/mask semantics changed"
    )


def test_hparams_from_config_defaults_to_gtrxl_128():
    """Table 4 GTrXL column with M=128, i.e. config_pure/craftax/gtrxl128.yaml."""
    hp = gxp.hparams_from_config({"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 512})
    assert (hp["layer_num"], hp["embedding_dim"], hp["head_dim"], hp["head_num"]) == (4, 128, 64, 4)
    assert (hp["mlp_num"], hp["memory_len"]) == (2, 128)
    assert hp["out_dim"] is None
    assert gxp.carry_size(hp) == 5 * 128 * 128 + 128 == 82048


@pytest.mark.parametrize("key", ["GX_LAYER_NUM", "GX_EMBEDDING_DIM", "GX_MEMORY_LEN", "GX_HEAD_NUM"])
def test_hparams_reject_nonpositive(key):
    with pytest.raises(ValueError):
        gxp.hparams_from_config({"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 512, key: 0})


def test_gradients_are_finite():
    model, params, carry, ins, resets = _build()

    def loss(p):
        return jnp.sum(model.apply(p, carry, (ins, resets))[1] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


def test_vmap_over_agents():
    n_agents, hp = 3, dict(SMALL)
    model = gxp.ScannedGTrXL(**hp)
    cs = gxp.carry_size(hp)
    carry = gxp.ScannedGTrXL.initialize_carry(B, cs, hp["memory_len"])
    ins = jnp.ones((T, B, hp["embedding_dim"]))
    resets = jnp.zeros((T, B))
    rngs = jax.random.split(jax.random.PRNGKey(1), n_agents)
    params = jax.vmap(lambda k: model.init(k, carry, (ins, resets))["params"])(rngs)
    out_carry, y = jax.vmap(lambda p, h, i, dn: model.apply({"params": p}, h, (i, dn)))(
        params,
        jnp.tile(carry[None], (n_agents, 1, 1)),
        jnp.tile(ins[None], (n_agents, 1, 1, 1)),
        jnp.tile(resets[None], (n_agents, 1, 1)),
    )
    assert y.shape == (n_agents, T, B, hp["embedding_dim"])
    assert out_carry.shape == (n_agents, B, cs)


# ---------------------------------------------------------------------------
# Equivalence with the authors' released implementation (opt-in).
# ---------------------------------------------------------------------------
_REF = os.environ.get("AGALITE_REF_DIR")


@pytest.mark.skipif(not _REF, reason="set AGALITE_REF_DIR to github.com/subho406/agalite")
def test_matches_reference_implementation():
    from flax.traverse_util import flatten_dict, unflatten_dict

    sys.path.insert(0, _REF)
    from src_pure.models.gtrxl import BatchedGTrXL  # noqa: E402

    L, D, DH, HN, MLP, M = 3, 16, 8, 2, 2, 6
    t, b = 5, 4
    hp = dict(layer_num=L, embedding_dim=D, head_dim=DH, head_num=HN, mlp_num=MLP, memory_len=M)
    mine = gxp.ScannedGTrXL(**hp)
    ref = BatchedGTrXL(head_dim=DH, embedding_dim=D, head_num=HN, mlp_num=MLP,
                       layer_num=L, memory_len=M)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    ins = jax.random.normal(k1, (t, b, D))
    resets = (jax.random.uniform(k2, (t, b)) < 0.25).astype(jnp.float32)
    my_carry = gxp.ScannedGTrXL.initialize_carry(b, gxp.carry_size(hp), M)
    ref_carry = BatchedGTrXL.initialize_carry(b, M, D, L)
    my_params = mine.init(k3, my_carry, (ins, resets))["params"]

    flat, out, P = flatten_dict(my_params), {}, ("VmapGTrXL_0",)
    for leaf in ("kernel", "bias"):
        out[P + ("embedding", "layers_0", leaf)] = flat[("embedding", leaf)]
    out[P + ("u",)], out[P + ("v",)] = flat[("u",)], flat[("v",)]
    for i in range(L):
        src, dst = f"layer_{i}", f"layers_{i}"
        for sub in ("attention_kv", "attention_q", "project", "project_pos"):
            for leaf in ("kernel", "bias"):
                out[P + (dst, "attention", sub, leaf)] = flat[(src, "attention", sub, leaf)]
        for g in ("gate1", "gate2"):
            out[P + (dst, g, "bgp")] = flat[(src, g, "bgp")]
            for w in ("Wr", "Ur", "Wz", "Uz", "Wg", "Ug"):
                out[P + (dst, g, w, "kernel")] = flat[(src, g, w, "kernel")]
        for ln in ("layernorm1", "layernorm2"):
            for leaf in ("scale", "bias"):
                out[P + (dst, ln, leaf)] = flat[(src, ln, leaf)]
        for j in range(MLP):
            for leaf in ("kernel", "bias"):
                out[P + (dst, "mlp", f"layers_{2 * j}", "layers_0", leaf)] = \
                    flat[(src, f"mlp_{j}", leaf)]
    ref_params = unflatten_dict(out)

    want = jax.eval_shape(lambda: ref.init(k3, ref_carry, (ins, resets))["params"])
    assert {k: v.shape for k, v in flatten_dict(ref_params).items()} == \
           {k: v.shape for k, v in flatten_dict(want).items()}

    my_new, my_y = mine.apply({"params": my_params}, my_carry, (ins, resets))
    ref_new, ref_y = ref.apply({"params": ref_params}, ref_carry, (ins, resets))
    np.testing.assert_allclose(np.asarray(my_y), np.asarray(ref_y), rtol=0, atol=1e-4)

    ref_mem, ref_mask = ref_new
    mine_mem = my_new[:, :gxp.memory_size(hp)].reshape(b, L + 1, M, D)
    np.testing.assert_allclose(np.asarray(mine_mem), np.asarray(ref_mem.squeeze(3)),
                               rtol=0, atol=1e-4)
    assert bool(jnp.all((my_new[:, gxp.memory_size(hp):] > 0.5) == ref_mask))

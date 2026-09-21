import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest
import yaml

import checkpoint_utils as ckpt
import pruning_utils as pruning

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1", reason="set RUN_E2E=1 to run the subprocess end-to-end test"
)

BASELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASELINES_DIR, "config", "test_checkpointing.yaml")


# The runs below restart after block 2 (update 4), so these two settings put the
# prune on either side of the restart: leg 2 still has to prune, or leg 1 already did.
PRUNE_AFTER_RESTART = {"SPARSE_ALG": "magnitude", "SPARSITY": 0.5, "PRUNE_STEP": 5}
PRUNE_BEFORE_RESTART = {"SPARSE_ALG": "magnitude", "SPARSITY": 0.5, "PRUNE_STEP": 1}


def _masks(carry):
    return carry[0][0].opt_state.masks


def _opt_count(carry):
    return int(carry[0][0].opt_state.count)


def _base_cfg(**overrides):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["WANDB_MODE"] = "disabled"
    cfg["SAVE_VIDEO"] = False
    cfg.update(overrides)
    return cfg


def _derived(cfg):
    num_updates = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    num_logging_iters = num_updates // cfg["LOGGING_UPDATES_INTERVAL"]
    return num_updates, num_logging_iters


def _write_cfg(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _run(cfg_path):
    runner = textwrap.dedent(
        f"""
        import sys, yaml
        sys.path.insert(0, {BASELINES_DIR!r})
        import seperate_ippo_rnn as m
        with open({cfg_path!r}) as f:
            cfg = yaml.safe_load(f)
        m.single_run(cfg)
        """
    )
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    result = subprocess.run(
        [sys.executable, "-c", runner],
        env=env,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    if result.returncode != 0:
        raise AssertionError(f"run failed:\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr[-4000:]}")
    return result


def _final_carry(cfg, ckpt_dir):
    import seperate_ippo_rnn as m

    env = m.build_env(cfg)
    init_carry, _, _ = m.make_train(cfg, env)(jax.random.PRNGKey(cfg["SEED"]))
    mngr = ckpt.make_manager(ckpt_dir, cfg)
    carry = ckpt.restore_carry(mngr, init_carry)
    mngr.close()
    return carry


@pytest.mark.parametrize(
    "overrides",
    [{}, PRUNE_AFTER_RESTART, PRUNE_BEFORE_RESTART],
    ids=["no_prune", "prune_after_restart", "prune_before_restart"],
)
def test_cli_resume_matches_uninterrupted(tmp_path, overrides):
    dir_a = str(tmp_path / "a_ckpt")
    dir_b = str(tmp_path / "b_ckpt")

    num_updates, num_logging_iters = _derived(_base_cfg())

    cfg_a = _base_cfg(**overrides)
    cfg_a["CHECKPOINT_DIR"] = dir_a
    cfg_a["OUTPUT_DIR"] = str(tmp_path / "a_out")
    cfg_a["RESUME"] = False
    path_a = str(tmp_path / "cfg_a.yaml")
    _write_cfg(path_a, cfg_a)
    _run(path_a)

    k = max(1, num_logging_iters // 2)
    cfg_b1 = _base_cfg(**overrides)
    cfg_b1["CHECKPOINT_DIR"] = dir_b
    cfg_b1["OUTPUT_DIR"] = str(tmp_path / "b_out")
    cfg_b1["RESUME"] = "auto"
    cfg_b1["MAX_BLOCKS_THIS_RUN"] = k
    path_b1 = str(tmp_path / "cfg_b1.yaml")
    _write_cfg(path_b1, cfg_b1)
    _run(path_b1)

    mngr_b = ckpt.make_manager(dir_b, cfg_b1)
    assert mngr_b.latest_step() == k * cfg_b1["LOGGING_UPDATES_INTERVAL"]
    mngr_b.close()

    cfg_b2 = dict(cfg_b1)
    cfg_b2["MAX_BLOCKS_THIS_RUN"] = 0
    path_b2 = str(tmp_path / "cfg_b2.yaml")
    _write_cfg(path_b2, cfg_b2)

    # What leg 2 restores from disk, read before it runs. _final_carry adds
    # derived keys to the config it is given, so cfg_b2 is written out first.
    carry_mid = _final_carry(cfg_b1, dir_b)

    _run(path_b2)

    carry_a = _final_carry(cfg_a, dir_a)
    carry_b = _final_carry(cfg_b2, dir_b)

    assert int(carry_a[1]) == num_updates
    assert int(carry_b[1]) == num_updates

    leaves_a = jax.tree_util.tree_leaves(carry_a)
    leaves_b = jax.tree_util.tree_leaves(carry_b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        assert jnp.allclose(x, y, atol=1e-5, rtol=1e-5), "resumed run diverged from uninterrupted run"

    if not overrides:
        return

    resume_update = k * cfg_b1["LOGGING_UPDATES_INTERVAL"]
    pruned_in_leg_1 = overrides["PRUNE_STEP"] < resume_update
    assert int(carry_mid[1]) == resume_update
    assert ckpt.read_sidecar(dir_b)["pruning"] == pruning.pruning_settings(cfg_b2)

    # The optimizer step count carries across the restart, so the prune fires once
    # at the configured update rather than restarting its countdown in leg 2.
    per_update = cfg_b2["UPDATE_EPOCHS"] * cfg_b2["NUM_MINIBATCHES"]
    assert _opt_count(carry_mid) == resume_update * per_update
    assert _opt_count(carry_b) == num_updates * per_update

    mid_sparsity = pruning.mask_sparsity(_masks(carry_mid))
    if pruned_in_leg_1:
        # Leg 1 pruned: leg 2 must restore those exact masks, not re-prune.
        assert jnp.allclose(mid_sparsity, overrides["SPARSITY"], atol=0.02)
        for x, y in zip(jax.tree.leaves(_masks(carry_mid)), jax.tree.leaves(_masks(carry_b))):
            assert jnp.array_equal(x, y), "masks changed across the restart"
    else:
        assert jnp.all(mid_sparsity == 0.0), "pruned before the configured step"
        assert all(bool(jnp.all(m == 1)) for m in jax.tree.leaves(_masks(carry_mid)))

    for carry in (carry_a, carry_b):
        assert jnp.allclose(pruning.mask_sparsity(_masks(carry)), overrides["SPARSITY"], atol=0.02)
        # Pruned weights are still exactly zero at the end of the resumed run.
        flat_masks = dict(jax.tree_util.tree_flatten_with_path(_masks(carry))[0])
        for path, p in jax.tree_util.tree_flatten_with_path(carry[0][0].params)[0]:
            if path in flat_masks:
                assert bool(jnp.all(jnp.where(flat_masks[path] == 0, p, 0.0) == 0.0)), path

"""One-shot, per-agent weight pruning for agent-stacked params, built on jaxpruner.

Every agent's network is pruned independently (its own top-k and ERK layer
allocation), all at the same optimizer step, exactly once. Pruned weights are
held at zero for the rest of training by re-applying the masks after every
gradient step, as jaxpruner's post_gradient_update does.
"""
import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

# One-shot pruners only: jaxpruner's other algorithms update masks repeatedly
# (rigl, set, static_sparse) or need a masked forward pass (the *_ste variants).
PRUNING_ALGORITHMS = ("magnitude", "random", "saliency", "global_magnitude", "global_saliency")


class PruneState(NamedTuple):
    """Optimizer state wrapping the inner optimizer's state with pruning masks."""

    masks: Any  # uint8, stacked over agents on axis 0; None for leaves never pruned (e.g. biases)
    count: jax.Array  # optimizer steps taken so far
    inner_state: Any


def pruning_enabled(config):
    """Return True if the config asks for pruning."""
    return config.get("SPARSE_ALG", "no_prune") != "no_prune"


def prune_count(config):
    """Return the optimizer step at which pruning happens (the first minibatch of update PRUNE_STEP)."""
    return config["PRUNE_STEP"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]


def pruning_settings(config):
    """Return the pruning settings that must stay fixed across resumes, or None if pruning is off."""
    if not pruning_enabled(config):
        return None
    return {
        "sparse_alg": config["SPARSE_ALG"],
        "sparsity": float(config["SPARSITY"]),
        "prune_step": int(config["PRUNE_STEP"]),
    }


def validate_config(config):
    """Raise ValueError on unusable pruning settings. Requires config["NUM_UPDATES"]."""
    if not pruning_enabled(config):
        return
    alg = config["SPARSE_ALG"]
    if alg not in PRUNING_ALGORITHMS:
        raise ValueError(
            f"SPARSE_ALG must be 'no_prune' or one of {PRUNING_ALGORITHMS}, got {alg!r}."
        )
    sparsity = config.get("SPARSITY")
    if not isinstance(sparsity, (int, float)) or not 0.0 < sparsity < 1.0:
        raise ValueError(f"SPARSITY must be a number in (0, 1) when pruning, got {sparsity!r}.")
    prune_step = config.get("PRUNE_STEP")
    if not isinstance(prune_step, int) or isinstance(prune_step, bool) or not 0 <= prune_step < config["NUM_UPDATES"]:
        raise ValueError(
            f"PRUNE_STEP must be an integer update step in [0, NUM_UPDATES={config['NUM_UPDATES']}), "
            f"got {prune_step!r}; otherwise pruning would never happen."
        )


def make_updater(config):
    """Create the jaxpruner updater for SPARSE_ALG and SPARSITY with an ERK layer allocation."""
    import jaxpruner
    from ml_collections import ConfigDict

    sparsity_config = ConfigDict()
    sparsity_config.algorithm = config["SPARSE_ALG"]
    sparsity_config.sparsity = config["SPARSITY"]
    sparsity_config.dist_type = "erk"
    # No update_start_step: jaxpruner's own scheduler is unused, as
    # wrap_optax_per_agent decides when to prune.
    return jaxpruner.create_updater_from_config(sparsity_config)


def _agent_slice(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def wrap_optax_per_agent(inner, updater, num_agents, prune_at, seed):
    """Wrap an optax transformation over agent-stacked params with one-shot per-agent pruning.

    At optimizer step prune_at, each agent's masks are computed by jaxpruner from
    that agent's params (and raw gradients, for saliency), before the step's
    update is applied; afterwards the masks never change. Callers must zero the
    pruned weights after each update with apply_masks.
    """
    # Distinct keys so random pruning does not give every agent the same mask.
    base_key = jax.random.PRNGKey(seed)
    agent_updaters = [
        dataclasses.replace(updater, rng_seed=jax.random.fold_in(base_key, i))
        for i in range(num_agents)
    ]

    def init_fn(params):
        # All-ones masks on the leaves jaxpruner prunes (weight matrices), None elsewhere.
        agent_masks = updater.init_state(_agent_slice(params, 0)).masks
        masks = jax.tree.map(lambda m: jnp.broadcast_to(m, (num_agents,) + m.shape), agent_masks)
        return PruneState(masks=masks, count=jnp.zeros([], jnp.int32), inner_state=inner.init(params))

    def compute_masks(params, grads):
        per_agent = []
        for i, agent_updater in enumerate(agent_updaters):
            agent_params = _agent_slice(params, i)
            agent_grads = _agent_slice(grads, i)
            fresh_state = agent_updater.init_state(agent_params)
            per_agent.append(agent_updater.update_state(fresh_state, agent_params, agent_grads).masks)
        return jax.tree.map(lambda *m: jnp.stack(m), *per_agent)

    def update_fn(updates, state, params=None):
        masks = jax.lax.cond(
            state.count == prune_at,
            compute_masks,
            lambda *_: state.masks,
            params,
            updates,
        )
        updates, inner_state = inner.update(updates, state.inner_state, params)
        return updates, PruneState(masks=masks, count=optax.safe_int32_increment(state.count), inner_state=inner_state)

    return optax.GradientTransformation(init_fn, update_fn)


def apply_masks(params, masks):
    """Zero the pruned weights. A no-op before pruning, while the masks are all ones."""
    return jax.tree.map(lambda p, m: p if m is None else p * m.astype(p.dtype), params, masks)


def mask_sparsity(masks):
    """Return each agent's fraction of prunable weights that are pruned, shape (num_agents,)."""
    leaves = jax.tree.leaves(masks)
    # Count pruned entries rather than taking 1 - kept / total, which XLA's
    # reciprocal-multiply rewrite leaves slightly above 0 before pruning.
    pruned = sum((m == 0).reshape(m.shape[0], -1).sum(axis=1, dtype=jnp.float32) for m in leaves)
    total = sum(m[0].size for m in leaves)
    return pruned / total

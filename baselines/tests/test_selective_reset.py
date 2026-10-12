import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.wrappers.baselines import LogWrapper

from craftax.craftax_env import make_craftax_env_from_name
from craftax.environment_base.wrappers import SelectiveResetVecEnvWrapper

if not hasattr(jax, "tree_map"):  # jaxmarl's MultiAgentEnv.step (the reference) still calls jax.tree_map
    jax.tree_map = jax.tree_util.tree_map

NUM_ENVS = 6
MAX_RESETS_PER_STEP = 2
EPISODE_CAP = 5
NUM_STEPS = 9


def _set_episode_cap(state):
    caps = jnp.full_like(state.env_state.effective_max_timesteps, EPISODE_CAP)
    return state.replace(env_state=state.env_state.replace(effective_max_timesteps=caps))


def test_selective_reset_matches_vmapped_auto_reset():
    """Bit-identical to jax.vmap(LogWrapper(env).step), which resets every env every step."""
    env = make_craftax_env_from_name("Craftax-Coop-Symbolic", num_teams=1, team_composition=(1, 1, 2))
    vec_env = SelectiveResetVecEnvWrapper(env, max_resets_per_step=MAX_RESETS_PER_STEP)
    reference_step = jax.jit(jax.vmap(LogWrapper(env).step))
    selective_step = jax.jit(vec_env.step)

    _, state = vec_env.reset(jax.random.split(jax.random.PRNGKey(0), NUM_ENVS))
    # Envs 5, 4, 3 finish one at a time, then no env, then envs 0-2 together (more than
    # MAX_RESETS_PER_STEP at once), so steps with 0, 1 and 3 resets all occur.
    timesteps = jnp.array([0, 0, 0, 2, 3, 4], dtype=state.env_state.timestep.dtype)
    state = _set_episode_cap(state.replace(env_state=state.env_state.replace(timestep=timesteps)))
    reference_state = selective_state = state

    rng = jax.random.PRNGKey(1)
    resets_per_step = []
    for _ in range(NUM_STEPS):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action_rngs = jax.random.split(action_rng, env.num_agents)
        actions = {
            agent: jax.random.randint(action_rngs[i], (NUM_ENVS,), 0, 25)
            for i, agent in enumerate(env.agents)
        }
        step_keys = jax.random.split(step_rng, NUM_ENVS)

        reference = reference_step(step_keys, reference_state, actions)
        selective = selective_step(step_keys, selective_state, actions)
        reference_leaves = jax.tree_util.tree_leaves(reference)
        selective_leaves = jax.tree_util.tree_leaves(selective)
        assert len(reference_leaves) == len(selective_leaves)
        for a, b in zip(reference_leaves, selective_leaves):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        resets_per_step.append(int(reference[3]["__all__"].sum()))
        reference_state = _set_episode_cap(reference[1])
        selective_state = _set_episode_cap(selective[1])

    assert 0 in resets_per_step
    assert any(1 <= n <= MAX_RESETS_PER_STEP for n in resets_per_step)
    assert any(n > MAX_RESETS_PER_STEP for n in resets_per_step)


def test_reset_pool_hands_out_pool_states_and_leaves_other_envs_alone():
    """step_with_reset_pool: envs that did not finish match step exactly; finished envs (in env
    order) take the next unused pool entries; the pool is regenerated when it runs short."""
    env = make_craftax_env_from_name("Craftax-Coop-Symbolic", num_teams=1, team_composition=(1, 1, 2))
    vec_env = SelectiveResetVecEnvWrapper(env, max_resets_per_step=MAX_RESETS_PER_STEP)
    plain_step = jax.jit(vec_env.step)
    pool_step = jax.jit(vec_env.step_with_reset_pool)

    _, state = vec_env.reset(jax.random.split(jax.random.PRNGKey(0), NUM_ENVS))
    timesteps = jnp.array([0, 0, 0, 2, 3, 4], dtype=state.env_state.timestep.dtype)
    state = _set_episode_cap(state.replace(env_state=state.env_state.replace(timestep=timesteps)))
    pool = vec_env.init_reset_pool(jax.random.PRNGKey(5), NUM_ENVS)  # small, so it gets refilled

    rng = jax.random.PRNGKey(1)
    refills = 0
    for _ in range(NUM_STEPS):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action_rngs = jax.random.split(action_rng, env.num_agents)
        actions = {
            agent: jax.random.randint(action_rngs[i], (NUM_ENVS,), 0, 25)
            for i, agent in enumerate(env.agents)
        }
        step_keys = jax.random.split(step_rng, NUM_ENVS)

        plain = plain_step(step_keys, state, actions)
        *pooled, new_pool = pool_step(step_keys, state, actions, pool)
        finished = np.asarray(plain[3]["__all__"])

        # Rewards, dones and info come from the step itself.
        for a, b in zip(jax.tree_util.tree_leaves(plain[2:]), jax.tree_util.tree_leaves(pooled[2:])):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        # Which pool the finished envs drew from, and from which entry.
        refilled = int(pool.next_index) + finished.sum() > NUM_ENVS
        source = vec_env.init_reset_pool(pool.key, NUM_ENVS) if refilled else pool
        first_entry = 0 if refilled else int(pool.next_index)
        refills += int(refilled)
        assert int(new_pool.next_index) == first_entry + finished.sum()

        entry = first_entry
        for i in range(NUM_ENVS):
            if finished[i]:
                expected = jax.tree_util.tree_map(lambda x: x[entry], (source.obs, source.env_state))
                entry += 1
            else:
                expected = jax.tree_util.tree_map(lambda x: x[i], (plain[0], plain[1].env_state))
            actual = jax.tree_util.tree_map(lambda x: x[i], (pooled[0], pooled[1].env_state))
            for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        state, pool = _set_episode_cap(pooled[1]), new_pool

    assert refills >= 1

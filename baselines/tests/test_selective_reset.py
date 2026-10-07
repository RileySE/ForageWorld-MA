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

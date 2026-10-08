from math import ceil, sqrt, floor

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from jaxmarl.wrappers.baselines import LogWrapper as JaxMARLLogWrapper
from matplotlib import pyplot as plt, animation



class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env: environment.Environment, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


# Wrapper to restrict to only the first 17 actions (the first floor stuff, basically)
class ReduceActionSpaceWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)
        self.action_space().n = 17

# Wrapper to add the action to the observation vector. Simple(?)
class AppendActionToObsWrapper(GymnaxWrapper):

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
        ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, env_state, reward, done, info = self._env.step(key, state, action, params)

        obs = jnp.concatenate([obs, action.reshape(action.shape + (1,))], axis=0)

        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:

        obs, env_state = self._env.reset(key, params)
        obs = jnp.concatenate([obs, jnp.zeros((1,))], axis=0)

        return obs, env_state



class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)

        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


def add_logging_fields(info, env_state):
    """Return `info` plus the per-step fields written to the logging CSVs (health, food, mob distances, ...).

    `env_state` is a single env's (unbatched) state; vmap over envs for batched states.
    """
    info = dict(info)

    # Add fields to be logged
    #info['action'] = jnp.concatenate(list(action.values()), axis=0)
    info['health'] = env_state.player_health
    info['food'] = env_state.player_food
    info['drink'] = env_state.player_drink
    info['energy'] = env_state.player_energy
    #info['done'] = done
    info['is_sleeping'] = env_state.is_sleeping
    info['is_resting'] = env_state.is_resting
    info['player_position_x'] = env_state.player_position[:,0]
    info['player_position_y'] = env_state.player_position[:,1]
    info['recover'] = env_state.player_recover
    info['hunger'] = env_state.player_hunger
    info['thirst'] = env_state.player_thirst  # print("A " + str(type(log_state)))
    info['fatigue'] = env_state.player_fatigue
    info['light_level'] = env_state.light_level
    # TODO why is this an array? It's supposed to be an int...
    #info['episode_id'] = env_state.env_id.squeeze()

    melee_pos = env_state.melee_mobs.position[env_state.level_index]
    melee_mask = env_state.melee_mobs.mask[env_state.level_index]
    passive_pos = env_state.passive_mobs.position[env_state.level_index]
    passive_mask = env_state.passive_mobs.mask[env_state.level_index]
    ranged_pos = env_state.ranged_mobs.position[env_state.level_index]
    ranged_mask = env_state.ranged_mobs.mask[env_state.level_index]

    # Per-player mob distance / count metrics (multi-agent adapted)
    # player_position: (num_players, 2), mob_pos: (num_mobs, 2), mob_mask: (num_mobs,)
    nearby_distance = 9

    def _mob_metrics_single_player(player_pos, mob_pos, mob_mask):
        """Compute closest-mob distance, on-screen flag, and nearby count for one player."""
        if mob_pos.shape[0] == 0:
            return (
                jnp.asarray(jnp.inf, dtype=jnp.float32),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
            )
        dists = jnp.linalg.norm(player_pos - mob_pos, ord=1, axis=-1)  # (num_mobs,)
        dists = jnp.where(mob_mask, dists, jnp.inf)
        closest_idx = jnp.argmin(dists)
        closest_dist_xy = player_pos - mob_pos[closest_idx]
        on_screen = jnp.logical_and(jnp.abs(closest_dist_xy[0]) <= 5,
                                    jnp.abs(closest_dist_xy[1]) <= 4)
        on_screen = jnp.logical_and(on_screen, mob_mask[closest_idx])
        dist = dists[closest_idx]
        num_nearby = (dists <= nearby_distance).sum()
        return dist, on_screen, num_nearby

    # vmap over players: player_pos axis 0, mob arrays broadcast (None)
    _mob_metrics_all_players = jax.vmap(_mob_metrics_single_player, in_axes=(0, None, None))

    dist_to_melee, melee_on_screen, num_melee_nearby = _mob_metrics_all_players(
        env_state.player_position, melee_pos, melee_mask)
    dist_to_passive, passive_on_screen, num_passives_nearby = _mob_metrics_all_players(
        env_state.player_position, passive_pos, passive_mask)
    if ranged_pos.shape[0] > 0:
        dist_to_ranged, ranged_on_screen, num_ranged_nearby = _mob_metrics_all_players(
            env_state.player_position, ranged_pos, ranged_mask)
    else:
        num_players = env_state.player_position.shape[0]
        dist_to_ranged = jnp.full((num_players,), jnp.inf)
        ranged_on_screen = jnp.zeros((num_players,), dtype=bool)
        num_ranged_nearby = jnp.zeros((num_players,), dtype=jnp.int32)

    num_monsters_killed = env_state.monsters_killed[env_state.level_index]

    info['dist_to_melee_l1'] = dist_to_melee
    info['melee_on_screen'] = melee_on_screen.astype(jnp.float32)
    info['dist_to_passive_l1'] = dist_to_passive
    info['passive_on_screen'] = passive_on_screen.astype(jnp.float32)
    info['dist_to_ranged_l1'] = dist_to_ranged.astype(jnp.float32)
    info['ranged_on_screen'] = ranged_on_screen.astype(jnp.float32)
    info['num_melee_nearby'] = num_melee_nearby.astype(jnp.float32)
    info['num_passives_nearby'] = num_passives_nearby.astype(jnp.float32)
    info['num_ranged_nearby'] = num_ranged_nearby.astype(jnp.float32)
    info['num_monsters_killed'] = num_monsters_killed
    info['has_sword'] = env_state.inventory.sword
    info['has_pick'] = env_state.inventory.pickaxe
    info['bow'] = env_state.inventory.bow
    info['arrows'] = env_state.inventory.arrows
    info['held_iron'] = env_state.inventory.iron

    return info


# Wrapper for plotting videos (every expensive op due to CPU latency, only run with this wrapper rarely!
class VideoPlotWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment, output_path='./', frames_per_file=500, do_videos=True):
        super().__init__(env)
        self.vis_renderer = None
        self.curr_env_id = -9999
        self.n_frames_seen = 0
        self.output_path = output_path
        self.frames_per_file = frames_per_file
        self.do_videos = do_videos

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key)

        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action)

        # Video plotting stuff
        # This needs to leave the jax ecosystem so we use callback
        def callback_func(new_obs, t, done):
            #if self.do_videos:
            self.vis_renderer.add_frame(new_obs, t, done)

        info = add_logging_fields(info, state.env_state)

        return obs, state, reward, done, info


class _NoAutoResetEnv(object):
    """Expose a jaxmarl env's step_env as `step`: the key is split exactly as in
    MultiAgentEnv.step, but the unconditional auto-reset is skipped."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, key, state, actions):
        key, _ = jax.random.split(key)
        return self._env.step_env(key, state, actions)


class SelectiveResetVecEnvWrapper(object):
    """Batched equivalent of jax.vmap(jaxmarl LogWrapper(env).step) that only resets finished envs.

    jaxmarl's MultiAgentEnv.step generates a fresh world for every env on every step and then
    discards it unless that env's episode just ended; for Craftax-Coop this world generation is
    about half the cost of a step. This wrapper steps all envs without resetting, then resets only
    the envs whose episode ended (at most `max_resets_per_step` per world-generation call), using
    the same reset key MultiAgentEnv.step would have used. Outputs are bit-identical to the
    vmapped original.

    Keys, states, actions and outputs all carry a leading num_envs axis. Under an outer vmap
    (e.g. over seeds) results stay correct, but the reset loop runs for every batch member.
    """

    def __init__(self, env, max_resets_per_step=8):
        self._env = env
        self._log_env = JaxMARLLogWrapper(_NoAutoResetEnv(env))
        self.max_resets_per_step = max_resets_per_step

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, keys):
        return jax.vmap(self._log_env.reset)(keys)

    def step(self, keys, state, actions):
        obs, state, reward, done, info = jax.vmap(self._log_env.step)(keys, state, actions)
        # MultiAgentEnv.step resets with key_reset from `key, key_reset = jax.random.split(key)`.
        reset_keys = jax.vmap(lambda key: jax.random.split(key)[1])(keys)
        obs, env_state = self._reset_finished(reset_keys, done["__all__"], obs, state.env_state)
        return obs, state.replace(env_state=env_state), reward, done, info

    def _reset_finished(self, reset_keys, finished, obs, env_state):
        num_envs = finished.shape[0]
        chunk_size = min(self.max_resets_per_step, num_envs)

        def has_pending(carry):
            return carry[0].any()

        def reset_chunk(carry):
            pending, obs, env_state = carry
            # Up to chunk_size pending env indices; padding points past the end and is dropped.
            idx = jnp.nonzero(pending, size=chunk_size, fill_value=num_envs)[0]
            new_obs, new_env_state = jax.vmap(self._env.reset)(
                reset_keys[jnp.minimum(idx, num_envs - 1)]
            )
            put = lambda old, new: old.at[idx].set(new, mode="drop")
            return (
                pending.at[idx].set(False, mode="drop"),
                jax.tree_util.tree_map(put, obs, new_obs),
                jax.tree_util.tree_map(put, env_state, new_env_state),
            )

        _, obs, env_state = jax.lax.while_loop(has_pending, reset_chunk, (finished, obs, env_state))
        return obs, env_state


# Class to progressively render visualization frames during test rollouts.
# TODO remove residual cruft
class VisualizationRenderer(object):
    # Set up plotting
    def __init__(self, frame_shape, save_path, enumerator, is_rgb=False, draw_only_first=False, frames_per_file=500):
        self.frame_shape = frame_shape
        self.save_path = save_path
        self.enumerator = enumerator
        self.is_rgb = is_rgb
        self.draw_only_first = draw_only_first
        self.frames_per_file = frames_per_file

        # frame_shape should be shaped like <x, y, n_channels>

        # Simple grid layout: n-by-n grid, possibly underfull
        self.side_length = 1

        # Determine figure aspect ratio
        self.obs_x = frame_shape[0]
        self.obs_y = frame_shape[1]

        self.fig = plt.figure(figsize=(ceil(self.obs_y / 100), ceil(self.obs_x / 100)))
        self.axs = self.fig.subplots(self.side_length, self.side_length, squeeze=(not draw_only_first))

        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in each frame
        self.ims = []

        self.n_frames_logged = 0
        self.last_timestep = 0
        self.n_videos_logged = 0
        self.key = None
        self.add_frame_callcount = 0

    # Render a new frame
    # Frames should have the shape described in frame_shape, with the first dimension being (usually) 1
    def add_frame(self, frame, timestep, done):

        self.add_frame_callcount += 1

        # If we finished, grab a different episode to log
        if self.add_frame_callcount % 1000000 == 0:
            self.key = None

        if not self.key:
            self.key = timestep
        # Attempt to log only one parallel env
        if timestep != self.key:
            return

        if self.n_frames_logged % 50 == 0:
            print('Logged', self.n_frames_logged, 'frames')
        #self.last_timestep = timestep
        curr_artist = []
        frame = frame / 255.
        # Draw the frame!
        im = self.axs.imshow(frame, animated=True, vmin=0, vmax=1)
        self.axs.set_xticks([])
        self.axs.set_yticks([])
        curr_artist.append(im)
        self.ims.append(curr_artist)
        self.n_frames_logged += 1

        if self.n_frames_logged >= self.frames_per_file:
            self.flush_video()

    # Write out the rendered frames as an mp4 using ffmpeg
    def flush_video(self):

        print('Flushing', len(self.ims), 'frames')
        # Animate/render the set of frames
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, blit=True,
                                        repeat_delay=1000, repeat=False)

        # Pipe to ffmpeg for encoding and writing to disk
        writer = animation.FFMpegWriter(
            fps=10, bitrate=-1, codec='hevc_nvenc')
        ani.save(self.save_path + "/example_episode_" + str(self.n_videos_logged) + ".mp4", writer=writer)

        plt.close()

        self.last_timestep = 0
        self.ims = []
        self.n_frames_logged = 0
        self.fig = plt.figure(figsize=(ceil(self.obs_y / 100), ceil(self.obs_x / 100)))
        self.axs = self.fig.subplots(self.side_length, self.side_length, squeeze=(not self.draw_only_first))
        self.n_videos_logged += 1


class CurriculumWrapper(GymnaxWrapper):

    def __init__(self, env: environment.Environment,
                 num_envs: int,
                 num_steps: int,
                 use_curriculum: bool,
                 predators: bool,
                 ):
        super().__init__(env)

        self.num_envs = num_envs
        self.total_steps = num_steps
        self.use_curriculum = use_curriculum
        self.predators = predators

        self.num_levels = 5

    def disable_predators(self, log_state):
        batched_max_melee_mobs = jnp.full((self.num_envs,), 0, dtype=jnp.int32)
        batched_max_ranged_mobs = jnp.full((self.num_envs,), 0, dtype=jnp.int32)

        env_state = log_state.env_state
        env_state = env_state.replace(max_melee_mobs=batched_max_melee_mobs,
                                      max_ranged_mobs=batched_max_ranged_mobs)
        return log_state.replace(env_state=env_state)

    def reset(
            self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, log_state = self._env.reset(rng, params)

        log_state = jax.lax.cond(
            not self.predators,
            lambda ls: self.disable_predators(log_state),
            lambda ls: log_state,
            log_state
        )

        return obs, log_state

    @partial(jax.jit, static_argnums=(0, 5))
    def step(
            self,
            key: chex.PRNGKey,
            log_state: LogEnvState,
            action: Union[int, float],
            update_step: int,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, log_state, reward, done, info = self._env.step(key, log_state, action, params)

        def update_curriculum(log_state, update_step):
            state = log_state.env_state

            # update level
            level = jnp.floor(update_step * self.num_levels / self.total_steps + 1).astype(jnp.int32)
            batched_level = jnp.full((self.num_envs,), level)
            state = state.replace(level=batched_level)

            # update spawn chances
            level_melee_spawn_chance = level / self.num_levels * state.max_melee_spawn_chance
            level_ranged_spawn_chance = level / self.num_levels * state.max_ranged_spawn_chance
            state = state.replace(max_melee_spawn_chance=level_melee_spawn_chance,
                                  max_ranged_spawn_chance=level_ranged_spawn_chance)

            # update max mobs
            max_melee_mobs = level / self.num_levels * 10
            max_ranged_mobs = level / self.num_levels * 10
            batched_max_melee_mobs = jnp.full((self.num_envs,), max_melee_mobs, dtype=jnp.int32)
            batched_max_ranged_mobs = jnp.full((self.num_envs,), max_ranged_mobs, dtype=jnp.int32)
            state = state.replace(max_melee_mobs=batched_max_melee_mobs,
                                  max_ranged_mobs=batched_max_ranged_mobs)

            return log_state.replace(env_state=state)

        log_state = jax.lax.cond(
            self.use_curriculum,
            lambda ls: update_curriculum(log_state, update_step),
            lambda ls: log_state,
            log_state
        )

        log_state = jax.lax.cond(
            not self.predators,
            lambda ls: self.disable_predators(log_state),
            lambda ls: log_state,
            log_state
        )


        return obs, log_state, reward, done, info

"""The faster mob code must match the code it replaced, bit for bit.

The sequential reference implementations below are copies of the original loops.
"""
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax_coop.constants import DIRECTIONS_PASSIVE, MOB_TYPE_COLLISION_MAPPING, OBS_DIM
from craftax_coop.craftax_state import Mobs
from craftax_coop.game_logic import move_passive_mobs
from craftax_coop.renderer.renderer_symbolic import add_mobs_to_obs_mob_map
from craftax_coop.util.game_logic_utils import (
    in_bounds,
    is_in_other_player,
    is_position_in_bounds_not_in_mob_not_colliding,
)
from craftax_coop.util.maths_utils import random_choice


def _assert_trees_equal(a, b):
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_random_choice_matches_jax_random_choice():
    keys = jax.random.split(jax.random.PRNGKey(0), 500)
    valid_passive_moves = jnp.arange(DIRECTIONS_PASSIVE.shape[0]) % 7 != 3  # bool, as in the env
    spawn_map = (jax.random.uniform(jax.random.PRNGKey(1), (96 * 96,)) < 0.01).astype(jnp.float32)
    cases = [
        (DIRECTIONS_PASSIVE, valid_passive_moves, ()),
        (jnp.arange(2), jnp.array([0.5, 0.5]), ()),
        (jnp.arange(2), jnp.array([1.0, 0.0]), ()),
        (jnp.arange(2), jnp.zeros(2) / jnp.zeros(2), ()),  # NaN, as for a 0/0 move preference
        (jnp.arange(4), jnp.zeros(4), ()),
        (jnp.arange(5), jnp.array([0.3, 0.3, 0.15, 0.125, 0.125]), ()),
        (jnp.arange(96 * 96), spawn_map / spawn_map.sum(), (1,)),
        (24, jnp.ones(24), ()),  # integer a means arange(a)
        (jnp.array([False, True]), jnp.array([0.9, 0.1]), (96, 96)),
    ]
    for a, p, shape in cases:
        expected = jax.vmap(lambda k: jax.random.choice(k, a, shape=shape, p=p))(keys)
        actual = jax.vmap(lambda k: random_choice(k, a, p=p, shape=shape))(keys)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _sequential_move_passive_mobs(rng, state, params, static_params):
    """The original per-mob scan from update_mobs."""

    def _move_passive_mob(rng_and_state, passive_mob_index):
        rng, state = rng_and_state
        passive_mobs = state.passive_mobs

        def _move_it(rng_and_state):
            rng, state = rng_and_state
            passive_mobs = state.passive_mobs
            rng, _rng = jax.random.split(rng)
            valid_random_moves = in_bounds(
                DIRECTIONS_PASSIVE + passive_mobs.position[state.player_level, passive_mob_index],
                static_params,
            )
            random_move_direction = jax.random.choice(_rng, DIRECTIONS_PASSIVE, p=valid_random_moves)
            proposed_position = (
                passive_mobs.position[state.player_level, passive_mob_index] + random_move_direction
            )
            mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
            collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
            valid_move = is_position_in_bounds_not_in_mob_not_colliding(
                state, proposed_position[None, :], collision_map, static_params
            )[0]
            in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
            valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_other_player))
            position = jax.lax.select(
                valid_move,
                proposed_position,
                passive_mobs.position[state.player_level, passive_mob_index],
            )
            return position, rng

        def _stay_static(rng_and_state):
            rng, state = rng_and_state
            return state.passive_mobs.position[state.player_level, passive_mob_index], rng

        position, rng = jax.lax.cond(params.passive_mobs_static, _stay_static, _move_it, rng_and_state)

        distance_to_players = jnp.abs(
            state.player_position - passive_mobs.position[state.player_level, passive_mob_index]
        ).sum(axis=1)
        should_not_despawn = jnp.logical_and(
            distance_to_players < params.mob_despawn_distance, state.player_alive
        ).any()
        old_position = state.passive_mobs.position[state.player_level, passive_mob_index]
        new_mob_map = state.mob_map.at[state.player_level, old_position[0], old_position[1]].set(
            jnp.logical_and(
                state.mob_map[state.player_level, old_position[0], old_position[1]],
                jnp.logical_not(passive_mobs.mask[state.player_level, passive_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.passive_mobs.mask[state.player_level, passive_mob_index], should_not_despawn
        )
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
        )
        state = state.replace(
            passive_mobs=state.passive_mobs.replace(
                position=state.passive_mobs.position.at[state.player_level, passive_mob_index].set(position),
                mask=state.passive_mobs.mask.at[state.player_level, passive_mob_index].set(new_mask),
            ),
            mob_map=new_mob_map,
        )
        return (rng, state), None

    (rng, state), _ = jax.lax.scan(
        _move_passive_mob, (rng, state), jnp.arange(static_params.max_passive_mobs)
    )
    return rng, state


def _crowd_passive_mobs(key, state, static_params):
    """Fill the passive-mob slots, mostly active, packed around player 0 so that moves collide."""
    level = state.player_level
    num_mobs = static_params.max_passive_mobs
    offset_key, mask_key = jax.random.split(key)
    offsets = jax.random.randint(offset_key, (num_mobs, 2), -4, 5)
    position = jnp.clip(state.player_position[0] + offsets, 0, static_params.map_size[0] - 1)
    mask = jax.random.uniform(mask_key, (num_mobs,)) < 0.85
    mobs = state.passive_mobs
    return state.replace(
        passive_mobs=mobs.replace(
            position=mobs.position.at[level].set(position),
            mask=mobs.mask.at[level].set(mask),
        ),
        mob_map=state.mob_map.at[level, position[:, 0], position[:, 1]].max(mask),
    )


def test_move_passive_mobs_matches_sequential_loop():
    env = make_craftax_env_from_name("Craftax-Coop-Symbolic", num_teams=1, team_composition=(1, 1, 2))
    static_params = env.static_env_params
    num_envs = 8
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2), num_envs))
    states = jax.vmap(_crowd_passive_mobs, in_axes=(0, 0, None))(
        jax.random.split(jax.random.PRNGKey(3), num_envs), states, static_params
    )
    rngs = jax.random.split(jax.random.PRNGKey(4), num_envs)
    for passive_mobs_static in (False, True):
        params = env.default_params.replace(passive_mobs_static=passive_mobs_static)
        expected = jax.jit(jax.vmap(lambda r, s: _sequential_move_passive_mobs(r, s, params, static_params)))(
            rngs, states
        )
        actual = jax.jit(jax.vmap(lambda r, s: move_passive_mobs(r, s, params, static_params)))(rngs, states)
        _assert_trees_equal(actual, expected)


def _sequential_add_mobs_to_obs_mob_map(mob_map, mobs, mob_class_index, player_position):
    """The original per-mob scan from render_craftax_symbolic."""
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_class_index = carry
        local_position = -1 * player_position + mobs.position[mob_index] + obs_dim_array // 2
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(axis=-1)
        on_screen *= mobs.mask[mob_index]
        mob_identifier = mob_class_index * 8 + mobs.type_id[mob_index]

        def _set_mobs_on_map(mob_map, local_position, on_screen):
            return mob_map.at[local_position[0], local_position[1], mob_identifier].set(
                on_screen.astype(jnp.int32)
            )

        mob_map = jax.vmap(_set_mobs_on_map, in_axes=(0, 0, 0))(mob_map, local_position, on_screen)
        return (mob_map, mobs, mob_class_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, mobs, mob_class_index), jnp.arange(mobs.mask.shape[0])
    )
    return mob_map


def test_obs_mob_map_matches_sequential_writes():
    num_players, num_mobs = 3, 40

    def random_case(key):
        keys = jax.random.split(key, 5)
        player_position = jax.random.randint(keys[0], (num_players, 2), 20, 76)
        # Mobs on screen, in the band where negative local coordinates wrap, and further away.
        position = player_position[0] + jax.random.randint(keys[1], (num_mobs, 2), -20, 21)
        mobs = Mobs(
            position=position,
            health=jnp.ones(num_mobs),
            mask=jax.random.uniform(keys[2], (num_mobs,)) < 0.7,
            attack_cooldown=jnp.zeros(num_mobs, dtype=jnp.int32),
            type_id=jax.random.randint(keys[3], (num_mobs,), 0, 8),
        )
        mob_map = jax.random.randint(keys[4], (num_players, *OBS_DIM, 40), 0, 2)
        return mob_map, mobs, player_position

    cases = jax.vmap(random_case)(jax.random.split(jax.random.PRNGKey(5), 32))
    for mob_class_index in range(5):
        expected = jax.vmap(
            lambda m, mobs, p: _sequential_add_mobs_to_obs_mob_map(m, mobs, mob_class_index, p)
        )(*cases)
        actual = jax.vmap(lambda m, mobs, p: add_mobs_to_obs_mob_map(m, mobs, mob_class_index, p))(*cases)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

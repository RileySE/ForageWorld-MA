"""The faster mob code must match the code it replaced, bit for bit.

The sequential reference implementations below are copies of the original loops.
"""
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax_coop.constants import (
    DIRECTIONS,
    DIRECTIONS_PASSIVE,
    MOB_TYPE_COLLISION_MAPPING,
    MOB_TYPE_DAMAGE_MAPPING,
    OBS_DIM,
    Achievement,
    BlockType,
    MobType,
)
from craftax_coop.craftax_state import Mobs
from craftax_coop.game_logic import (
    _move_mob_projectile,
    _move_player_projectile,
    _move_melee_mobs_with_keys,
    move_mob_projectiles,
    _move_passive_mobs_with_keys,
    move_player_projectiles,
)
from craftax_coop.renderer.renderer_symbolic import add_mobs_to_obs_mob_map
from craftax_coop.util.game_logic_utils import (
    get_damage_done_to_player,
    in_bounds,
    is_fighting_boss,
    is_in_other_player,
    is_position_in_bounds_not_in_mob_not_colliding,
)
from craftax_coop.util.maths_utils import random_choice


def _chained_passive_mob_keys(rng, num_mobs):
    """The keys the original passive-mob loop drew: one split of the carried key per mob."""
    keys = []
    for _ in range(num_mobs):
        rng, key = jax.random.split(rng)
        keys.append(key)
    return rng, jnp.stack(keys)


def _chained_melee_mob_keys(rng, num_mobs):
    """The keys the original melee-mob loop drew: four splits of the carried key per mob, the
    second half of the last one being carried to the next mob."""
    move_keys, axis_keys, chase_keys = [], [], []
    for _ in range(num_mobs):
        rng, move_key = jax.random.split(rng)
        rng, axis_key = jax.random.split(rng)
        rng, chase_key = jax.random.split(rng)
        _, rng = jax.random.split(rng)
        move_keys.append(move_key)
        axis_keys.append(axis_key)
        chase_keys.append(chase_key)
    return rng, jnp.stack(move_keys), jnp.stack(axis_keys), jnp.stack(chase_keys)


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
                DIRECTIONS_PASSIVE + passive_mobs.position[state.level_index, passive_mob_index],
                static_params,
            )
            random_move_direction = jax.random.choice(_rng, DIRECTIONS_PASSIVE, p=valid_random_moves)
            proposed_position = (
                passive_mobs.position[state.level_index, passive_mob_index] + random_move_direction
            )
            mob_type = passive_mobs.type_id[state.level_index, passive_mob_index]
            collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
            valid_move = is_position_in_bounds_not_in_mob_not_colliding(
                state, proposed_position[None, :], collision_map, static_params
            )[0]
            in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
            valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_other_player))
            position = jax.lax.select(
                valid_move,
                proposed_position,
                passive_mobs.position[state.level_index, passive_mob_index],
            )
            return position, rng

        def _stay_static(rng_and_state):
            rng, state = rng_and_state
            return state.passive_mobs.position[state.level_index, passive_mob_index], rng

        position, rng = jax.lax.cond(params.passive_mobs_static, _stay_static, _move_it, rng_and_state)

        distance_to_players = jnp.abs(
            state.player_position - passive_mobs.position[state.level_index, passive_mob_index]
        ).sum(axis=1)
        should_not_despawn = jnp.logical_and(
            distance_to_players < params.mob_despawn_distance, state.player_alive
        ).any()
        old_position = state.passive_mobs.position[state.level_index, passive_mob_index]
        new_mob_map = state.mob_map.at[state.level_index, old_position[0], old_position[1]].set(
            jnp.logical_and(
                state.mob_map[state.level_index, old_position[0], old_position[1]],
                jnp.logical_not(passive_mobs.mask[state.level_index, passive_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.passive_mobs.mask[state.level_index, passive_mob_index], should_not_despawn
        )
        new_mob_map = new_mob_map.at[state.level_index, position[0], position[1]].set(
            jnp.logical_or(new_mob_map[state.level_index, position[0], position[1]], new_mask)
        )
        state = state.replace(
            passive_mobs=state.passive_mobs.replace(
                position=state.passive_mobs.position.at[state.level_index, passive_mob_index].set(position),
                mask=state.passive_mobs.mask.at[state.level_index, passive_mob_index].set(new_mask),
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
    level = state.level_index
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
        expected_rng, expected_state = jax.jit(jax.vmap(
            lambda r, s: _sequential_move_passive_mobs(r, s, params, static_params)
        ))(rngs, states)
        chained_rng, mob_keys = jax.vmap(
            lambda r: _chained_passive_mob_keys(r, static_params.max_passive_mobs)
        )(rngs)
        actual_state = jax.jit(jax.vmap(
            lambda k, s: _move_passive_mobs_with_keys(k, s, params, static_params)
        ))(mob_keys, states)
        _assert_trees_equal(actual_state, expected_state)
        # The loop draws no keys for static mobs.
        _assert_trees_equal(rngs if passive_mobs_static else chained_rng, expected_rng)


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


def _sequential_move_melee_mobs(rng, state, params, static_params):
    """The original per-mob scan from update_mobs."""

    def _move_melee_mob(rng_and_state, melee_mob_index):
        rng, state = rng_and_state
        melee_mobs = state.melee_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + melee_mobs.position[state.level_index, melee_mob_index],
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
            p=valid_random_moves
        )
        random_move_proposed_position = (
            melee_mobs.position[state.level_index, melee_mob_index]
            + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position
            - melee_mobs.position[state.level_index, melee_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(
            state.player_alive,
            distance_to_players,
            jnp.inf
        ))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]

        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_targetted, player_move_direction_index]
                - melee_mobs.position[state.level_index, melee_mob_index, player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            melee_mobs.position[state.level_index, melee_mob_index]
            + player_move_direction
        )

        # Choose movement
        close_to_player = distance_to_players < 10
        close_to_player = jnp.logical_and(
            close_to_player,
            state.player_alive
        ).any()
        close_to_player = jnp.logical_or(
            close_to_player, is_fighting_boss(state, static_params)
        )
        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(
            close_to_player, jax.random.uniform(_rng) < 0.75
        )
        proposed_position = jax.lax.select(
            close_to_player,
            player_move_proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = distance_to_players == 1
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            state.player_alive
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            melee_mobs.attack_cooldown[state.level_index, melee_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, melee_mobs.mask[state.level_index, melee_mob_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player.any(),
            melee_mobs.position[state.level_index, melee_mob_index],
            proposed_position,
        )

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            melee_mobs.type_id[state.level_index, melee_mob_index], MobType.MELEE.value
        ]

        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping[:, None])
        )

        new_cooldown = jnp.where(
            is_attacking_player.any(),
            5,
            melee_mobs.attack_cooldown[state.level_index, melee_mob_index] - 1,
        )

        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

        melee_damage_taken = melee_mob_damage * is_attacking_player

        state = state.replace(
            player_health=state.player_health - melee_mob_damage * is_attacking_player,
            is_sleeping=jnp.logical_and(
                state.is_sleeping, jnp.logical_not(is_attacking_player)
            ),
            is_resting=jnp.logical_and(
                state.is_resting, jnp.logical_not(is_attacking_player)
            ),
            achievements=state.achievements.at[:, Achievement.WAKE_UP.value].set(
                jnp.logical_or(
                    state.achievements[:, Achievement.WAKE_UP.value], is_waking_player
                )
            ),
            damage_taken_melee=state.damage_taken_melee + melee_damage_taken,
            log_predator_hit=jnp.maximum(
                state.log_predator_hit,
                is_attacking_player.astype(state.log_predator_hit.dtype),
            ),
        )

        mob_type = melee_mobs.type_id[state.level_index, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )


        position = jax.lax.select(
            valid_move,
            proposed_position,
            melee_mobs.position[state.level_index, melee_mob_index],
        )

        # Melee despawn behavior depends on config flag
        # Passive mobs (including snails) use their own logic and are unchanged.
        should_not_despawn = jax.lax.select(
            params.melee_mobs_despawn_when_far,
            (distance_to_players < params.melee_mob_despawn_distance).any(),  # if flag=True. distance-gated and can despawn
            jnp.asarray(True), # if flag=False always persist
        )

        rng, _rng = jax.random.split(rng)

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.level_index,
            state.melee_mobs.position[state.level_index, melee_mob_index, 0],
            state.melee_mobs.position[state.level_index, melee_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.level_index,
                    state.melee_mobs.position[state.level_index, melee_mob_index, 0],
                    state.melee_mobs.position[state.level_index, melee_mob_index, 1],
                ],
                jnp.logical_not(melee_mobs.mask[state.level_index, melee_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.melee_mobs.mask[state.level_index, melee_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.level_index, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.level_index, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            melee_mobs=state.melee_mobs.replace(
                position=state.melee_mobs.position.at[
                    state.level_index, melee_mob_index
                ].set(position),
                attack_cooldown=state.melee_mobs.attack_cooldown.at[
                    state.level_index, melee_mob_index
                ].set(new_cooldown),
                mask=state.melee_mobs.mask.at[state.level_index, melee_mob_index].set(
                    new_mask
                ),
            ),
            mob_map=new_mob_map,
        )

        return (_rng, state), None

    (rng, state), _ = jax.lax.scan(
        _move_melee_mob, (rng, state), jnp.arange(static_params.max_melee_mobs)
    )
    return rng, state


def _crowd_melee_mobs(key, state, static_params):
    """Pack melee mobs around the players, most of them next to one, with mostly expired cooldowns,
    sleeping and dead players, and varied armour, so that several mobs attack the same player and
    wake-ups and collisions all occur."""
    level = state.level_index
    num_mobs = static_params.max_melee_mobs
    num_players = state.player_position.shape[0]
    keys = jax.random.split(key, 12)
    anchor = state.player_position[jax.random.randint(keys[0], (num_mobs,), 0, num_players)]
    nearby = anchor + jax.random.randint(keys[1], (num_mobs, 2), -3, 4)
    adjacent = anchor + DIRECTIONS[1:5][jax.random.randint(keys[10], (num_mobs,), 0, 4)]
    position = jnp.where((jax.random.uniform(keys[11], (num_mobs,)) < 0.6)[:, None], adjacent, nearby)
    position = jnp.clip(position, 0, static_params.map_size[0] - 1)
    mask = jax.random.uniform(keys[2], (num_mobs,)) < 0.8
    mobs = state.melee_mobs
    return state.replace(
        melee_mobs=mobs.replace(
            position=mobs.position.at[level].set(position),
            mask=mobs.mask.at[level].set(mask),
            attack_cooldown=mobs.attack_cooldown.at[level].set(
                jax.random.randint(keys[3], (num_mobs,), -1, 3)
            ),
        ),
        mob_map=state.mob_map.at[level, position[:, 0], position[:, 1]].max(mask),
        is_sleeping=jax.random.uniform(keys[4], (num_players,)) < 0.5,
        is_resting=jax.random.uniform(keys[5], (num_players,)) < 0.3,
        player_alive=jax.random.uniform(keys[6], (num_players,)) < 0.85,
        player_health=jax.random.uniform(keys[7], (num_players,), minval=0.5, maxval=9.0),
        inventory=state.inventory.replace(
            armour=jax.random.randint(keys[8], state.inventory.armour.shape, 0, 3)
        ),
        armour_enchantments=jax.random.randint(keys[9], state.armour_enchantments.shape, 0, 3),
    )


def test_move_melee_mobs_matches_sequential_loop():
    env = make_craftax_env_from_name("Craftax-Coop-Symbolic", num_teams=1, team_composition=(1, 1, 2))
    static_params = env.static_env_params
    num_envs = 8
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(6), num_envs))
    states = jax.vmap(_crowd_melee_mobs, in_axes=(0, 0, None))(
        jax.random.split(jax.random.PRNGKey(7), num_envs), states, static_params
    )
    rngs = jax.random.split(jax.random.PRNGKey(8), num_envs)
    for despawn_when_far in (False, True):
        params = env.default_params.replace(melee_mobs_despawn_when_far=despawn_when_far)
        expected = jax.jit(jax.vmap(
            lambda r, s: _sequential_move_melee_mobs(r, s, params, static_params)
        ))(rngs, states)
        chained_rng, move_keys, axis_keys, chase_keys = jax.vmap(
            lambda r: _chained_melee_mob_keys(r, static_params.max_melee_mobs)
        )(rngs)
        actual_state = jax.jit(jax.vmap(
            lambda a, b, c, s: _move_melee_mobs_with_keys(a, b, c, s, params, static_params)
        ))(move_keys, axis_keys, chase_keys, states)
        _assert_trees_equal((chained_rng, actual_state), expected)


def _sequential_slots(move_one, state, num_slots):
    """The original loop: move_one for every slot, in order."""
    state, _ = jax.lax.scan(lambda state, i: (move_one(state, i), None), state, jnp.arange(num_slots))
    return state


def _scatter_projectiles(key, state, static_params):
    """Projectiles (some active) near players, mobs, benches/furnaces/water and the map edge.

    Also gives some envs an active melee mob with health 0 (unreachable in play) to check that
    attack_mob's clean-up of such mobs still happens exactly where the per-slot loop did it.
    """
    level = state.level_index
    num_players = state.player_position.shape[0]
    keys = iter(jax.random.split(key, 40))
    centre = state.player_position[0]
    edge = static_params.map_size[0] - 1

    def projectile_slots(mobs, num_slots, owners_and_directions):
        # A quarter of envs have no active projectiles of this kind.
        active_fraction = jax.random.uniform(next(keys)) * (jax.random.uniform(next(keys)) < 0.75)
        near = jnp.clip(centre + jax.random.randint(next(keys), (num_slots, 2), -3, 4), 0, edge)
        on_edge = jnp.stack(
            [jax.random.choice(next(keys), jnp.array([0, edge]), (num_slots,)),
             jax.random.randint(next(keys), (num_slots,), 0, edge + 1)], axis=-1
        )
        position = jnp.where((jax.random.uniform(next(keys), (num_slots,)) < 0.25)[:, None], on_edge, near)
        mask = jax.random.uniform(next(keys), (num_slots,)) < active_fraction
        directions = DIRECTIONS[1:5][jax.random.randint(next(keys), (num_slots,), 0, 4)]
        owners = jax.random.randint(next(keys), (num_slots,), 0, num_players)
        mobs = mobs.replace(
            position=mobs.position.at[level].set(position),
            mask=mobs.mask.at[level].set(mask),
            type_id=mobs.type_id.at[level].set(jax.random.randint(next(keys), (num_slots,), 0, 8)),
        )
        dir_array, owner_array = owners_and_directions
        return mobs, dir_array.at[level].set(directions), owner_array.at[level].set(owners)

    mob_projectiles, mob_directions, mob_owners = projectile_slots(
        state.mob_projectiles, static_params.max_mob_projectiles,
        (state.mob_projectile_directions, state.mob_projectile_owners),
    )
    player_projectiles, player_directions, player_owners = projectile_slots(
        state.player_projectiles, static_params.max_player_projectiles,
        (state.player_projectile_directions, state.player_projectile_owners),
    )

    # Benches, furnaces, water and stone around the players.
    cells = jnp.clip(centre + jax.random.randint(next(keys), (30, 2), -4, 5), 0, edge)
    blocks = jnp.array([BlockType.CRAFTING_TABLE.value, BlockType.FURNACE.value, BlockType.WATER.value,
                        BlockType.STONE.value, BlockType.PATH.value])
    level_map = state.map[level].at[cells[:, 0], cells[:, 1]].set(
        blocks[jax.random.randint(next(keys), (30,), 0, len(blocks))]
    )

    # Weak melee and passive mobs near the players; maybe one active melee mob with health 0.
    def weak_mobs(mobs, mob_map):
        num_mobs = mobs.mask.shape[1]
        position = jnp.clip(centre + jax.random.randint(next(keys), (num_mobs, 2), -3, 4), 0, edge)
        mask = jax.random.uniform(next(keys), (num_mobs,)) < 0.5
        health = jax.random.choice(next(keys), jnp.array([0.5, 1.0, 3.0]), (num_mobs,))
        mobs = mobs.replace(
            position=mobs.position.at[level].set(position),
            mask=mobs.mask.at[level].set(mask),
            health=mobs.health.at[level].set(health),
        )
        return mobs, mob_map.at[level, position[:, 0], position[:, 1]].max(mask)

    melee_mobs, mob_map = weak_mobs(state.melee_mobs, state.mob_map)
    passive_mobs, mob_map = weak_mobs(state.passive_mobs, mob_map)
    zero_health_mob = jax.random.uniform(next(keys)) < 0.5
    melee_mobs = melee_mobs.replace(
        mask=melee_mobs.mask.at[level, 0].set(jnp.logical_or(melee_mobs.mask[level, 0], zero_health_mob)),
        health=melee_mobs.health.at[level, 0].set(jnp.where(zero_health_mob, 0.0, melee_mobs.health[level, 0])),
    )

    return state.replace(
        map=state.map.at[level].set(level_map),
        mob_map=mob_map,
        melee_mobs=melee_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_directions,
        mob_projectile_owners=mob_owners,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_directions,
        player_projectile_owners=player_owners,
        is_sleeping=jax.random.uniform(next(keys), (num_players,)) < 0.5,
        is_resting=jax.random.uniform(next(keys), (num_players,)) < 0.3,
        player_health=jax.random.uniform(next(keys), (num_players,), minval=0.5, maxval=9.0),
        bow_enchantment=jax.random.randint(next(keys), (num_players,), 0, 3),
        player_dexterity=jax.random.randint(next(keys), (num_players,), 1, 6),
        player_intelligence=jax.random.randint(next(keys), (num_players,), 1, 6),
    )


def test_projectile_moves_match_sequential_loop():
    # Two teams, so that projectiles can hit (cross-team) players.
    env = make_craftax_env_from_name("Craftax-Coop-Symbolic", num_teams=2, team_composition=(1, 1, 2))
    params, static_params = env.default_params, env.static_env_params
    num_envs = 16
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(9), num_envs))
    states = jax.vmap(_scatter_projectiles, in_axes=(0, 0, None))(
        jax.random.split(jax.random.PRNGKey(10), num_envs), states, static_params
    )

    move_mob = lambda state, i: _move_mob_projectile(state, i, static_params)
    expected = jax.jit(jax.vmap(
        lambda s: _sequential_slots(move_mob, s, static_params.max_mob_projectiles)))(states)
    actual = jax.jit(jax.vmap(lambda s: move_mob_projectiles(s, static_params)))(states)
    _assert_trees_equal(actual, expected)

    move_player = lambda state, i: _move_player_projectile(state, i, params, static_params)
    expected = jax.jit(jax.vmap(
        lambda s: _sequential_slots(move_player, s, static_params.max_player_projectiles)))(states)
    actual = jax.jit(jax.vmap(lambda s: move_player_projectiles(s, params, static_params)))(states)
    _assert_trees_equal(actual, expected)

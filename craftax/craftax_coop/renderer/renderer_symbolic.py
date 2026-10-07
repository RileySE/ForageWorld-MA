import jax
from functools import partial

from craftax_coop.constants import *
from craftax_coop.craftax_state import EnvState, StaticEnvParams
from craftax_coop.util.game_logic_utils import is_boss_vulnerable


OBS_MOB_TYPES_PER_CLASS = 8


def add_mobs_to_obs_mob_map(mob_map, mobs, mob_class_index, player_position):
    """Same result as writing the mobs into mob_map one at a time in index order.

    Each mob writes on_screen (0 or 1) into its class/type channel at its local cell in every
    player's view, so per (player, cell, channel) the last mob that writes wins. Off-screen and
    inactive mobs write 0, and negative local coordinates wrap around (numpy-style indexing),
    so such a mob can clear a cell that an earlier mob set; this existing behaviour is kept.
    """
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    local_position = (
        -1 * player_position[:, None, :]
        + mobs.position[None, :, :]
        + obs_dim_array // 2
    )  # (num_players, num_mobs, 2)
    on_screen = jnp.logical_and(
        local_position >= 0, local_position < obs_dim_array
    ).all(axis=-1)
    on_screen = jnp.logical_and(on_screen, mobs.mask[None, :])

    wrapped = jnp.where(local_position < 0, local_position + obs_dim_array, local_position)
    writes = jnp.logical_and(wrapped >= 0, wrapped < obs_dim_array).all(axis=-1)
    num_mobs = mobs.mask.shape[0]
    # Odd stamps write 1, even stamps write 0; the highest stamp per cell is the last write.
    stamp = jnp.where(writes, 2 * jnp.arange(num_mobs)[None, :] + on_screen, -1)
    wrapped = jnp.clip(wrapped, 0, obs_dim_array - 1)
    mob_identifier = mob_class_index * OBS_MOB_TYPES_PER_CLASS + mobs.type_id
    last_write = jnp.full(mob_map.shape, -1, dtype=jnp.int32).at[
        jnp.arange(mob_map.shape[0])[:, None],
        wrapped[..., 0],
        wrapped[..., 1],
        mob_identifier[None, :],
    ].max(stamp)
    return jnp.where(last_write >= 0, last_write % 2, mob_map)


def render_craftax_symbolic(state: EnvState, static_params: StaticEnvParams):
    map = state.map[state.player_level]

    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Items
    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    # Create item map view for each player
    item_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_items_map, tl_corner, OBS_DIM
    )
    item_map_view_one_hot = jax.nn.one_hot(item_map_view, num_classes=len(ItemType))

    # Mobs
    mob_map = jnp.zeros(
        (static_params.player_count, *OBS_DIM, 5 * OBS_MOB_TYPES_PER_CLASS), dtype=jnp.int32
    )  # 5 classes * 8 types

    mob_classes = [
        state.melee_mobs,
        state.passive_mobs,
        state.ranged_mobs,
        state.mob_projectiles,
        state.player_projectiles,
    ]
    for mob_class_index, mobs in enumerate(mob_classes):
        if mobs.mask.shape[1] > 0:
            mob_map = add_mobs_to_obs_mob_map(
                mob_map,
                jax.tree_util.tree_map(lambda x: x[state.player_level], mobs),
                mob_class_index,
                state.player_position,
            )

    def reorder_teammate_info(teammate_info, player_index):
        i1 = (jnp.arange(static_params.player_count) == 0) * player_index
        i2 = jnp.logical_and(jnp.arange(static_params.player_count) > 0, jnp.arange(static_params.player_count) <= (player_index)) * (jnp.arange(static_params.player_count) - 1)
        i3 = (jnp.arange(static_params.player_count) > player_index) * (jnp.arange(static_params.player_count))
        indices = (i1 + i2 + i3)
        return teammate_info[indices].flatten()
    
    # Teammate map (One-hot encoding of teammate + bit for dead/alive)
    def _add_teammate(player_index):
        """Creates teammate map for each player"""
        teammate_map = jnp.zeros(
            (*OBS_DIM, static_params.player_count + 1), dtype=jnp.int32
        )
        local_position = (
            -1 * state.player_position[player_index]
            + state.player_position
            + obs_dim_array // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)

        # Add teammate encoding
        teammate_map = teammate_map.at[
            local_position[:, 0], local_position[:, 1],
            (
                (jnp.arange(static_params.player_count) < player_index) * (jnp.arange(static_params.player_count) + 1) +
                (jnp.arange(static_params.player_count) == player_index) * 0 +
                (jnp.arange(static_params.player_count) > player_index) * (jnp.arange(static_params.player_count))
            )
        ].max(on_screen)

        # Add dead/alive bit
        teammate_map = teammate_map.at[
            local_position[:, 0], local_position[:, 1], -1
        ].set(
            jnp.logical_and(
                on_screen,
                state.player_alive
            )
        )

        """
        Find direction to teammates
        """
        direction_index_2d = jnp.where(
            local_position < 0, 1,
            jnp.where(local_position >= obs_dim_array, 2, 0)
        )
        direction_index = direction_index_2d[:, 0]*3 + direction_index_2d[:, 1] - 1
        teammate_directions = jax.nn.one_hot(direction_index, num_classes=8)
        teammate_directions = reorder_teammate_info(teammate_directions, player_index)
        return teammate_map, teammate_directions
    teammate_map, teammate_directions = jax.vmap(_add_teammate, in_axes=0)(jnp.arange(static_params.player_count))
    teammate_directions = teammate_directions * static_params.use_teammate_direction
        # return teammate_map 
    # teammate_map  = jax.vmap(_add_teammate, in_axes=0)(jnp.arange(static_params.player_count))
    # Concat all maps
    all_map = jnp.concatenate(
        [map_view_one_hot, item_map_view_one_hot, mob_map, teammate_map], axis=-1
    )

    # Light map
    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )

    # create light map for each player
    light_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_view = light_map_view > 0.05

    # Mask out tiles and mobs in darkness
    all_map = all_map * light_map_view[:, :, :, None]
    all_map = jnp.concatenate(
        (all_map, jnp.expand_dims(light_map_view, axis=-1)), axis=-1
    )

    # Inventory
    inventory = jnp.stack(
        (
            jnp.sqrt(state.inventory.wood) / 10.0,
            jnp.sqrt(state.inventory.stone) / 10.0,
            jnp.sqrt(state.inventory.coal) / 10.0,
            jnp.sqrt(state.inventory.iron) / 10.0,
            jnp.sqrt(state.inventory.diamond) / 10.0,
            jnp.sqrt(state.inventory.sapphire) / 10.0,
            jnp.sqrt(state.inventory.ruby) / 10.0,
            jnp.sqrt(state.inventory.sapling) / 10.0,
            jnp.sqrt(state.inventory.torches) / 10.0,
            jnp.sqrt(state.inventory.arrows) / 10.0,
            state.inventory.books,
            state.inventory.pickaxe / 4.0,
            state.inventory.sword / 4.0,
            state.sword_enchantment,
            state.bow_enchantment,
            state.inventory.bow,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    potions = jnp.sqrt(state.inventory.potions) / 10.0
    armour = state.inventory.armour / 2.0
    armour_enchantments = state.armour_enchantments

    intrinsics = jnp.stack(
        (
            # state.player_health / 10.0, # -- Removed and placed as part of the teammate dashboard
            state.player_food / 10.0,
            state.player_drink / 10.0,
            state.player_energy / 10.0,
            state.player_mana / 10.0,
            state.player_xp / 10.0,
            state.player_dexterity / 10.0,
            state.player_strength / 10.0,
            state.player_intelligence / 10.0,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    special_values_per_player = jnp.stack(
        (
            state.is_sleeping,
            state.is_resting,
            state.learned_spells,
        ),
        axis=1,
    )
    special_values_level = jnp.array(
        [
            state.light_level,
            state.player_level / 10.0,
            state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            is_boss_vulnerable(state),
        ]
    )

    """
    Teammate Dashboard (team-only)
        Includes:
            - Player Health
            - Player Dead or Alive
            - Specialization
            - Requested Material
    Only shows info about players on the SAME team.
    """
    players_health = state.player_health / 10.0
    players_alive = state.player_alive
    players_specialization = jax.nn.one_hot(state.player_specialization - Specialization.FORAGER.value, num_classes=3)
    request_matches = state.request_type[:, None] == REQUEST_ACTIONS[None, :]
    request_index = jnp.argmax(request_matches, axis=1)
    has_valid_request_type = request_matches.any(axis=1)
    requested_material = (
        jax.nn.one_hot(request_index, num_classes=REQUEST_ACTIONS.shape[0])
        * (state.request_duration > 0)[:, None]
        * has_valid_request_type[:, None]
    )
    player_data = jnp.concatenate(
        (players_health[:, None], players_alive[:, None], players_specialization, requested_material),
        axis=-1
    )

    agents_per_team = len(static_params.team_composition)

    def reorder_team_only_info(player_data, player_index):
        """Reorder so self is first, but only include same-team players."""
        team_id = player_index // agents_per_team
        team_start = team_id * agents_per_team
        team_indices = team_start + jnp.arange(agents_per_team)
        team_data = player_data[team_indices]
        within_team_idx = player_index - team_start
        idx = jnp.arange(agents_per_team)
        i1 = (idx == 0) * within_team_idx
        i2 = jnp.logical_and(idx > 0, idx <= within_team_idx) * (idx - 1)
        i3 = (idx > within_team_idx) * idx
        indices = i1 + i2 + i3
        return team_data[indices].flatten()

    teammate_dashboard = jax.vmap(lambda i: reorder_team_only_info(player_data, i))(jnp.arange(static_params.player_count))

    all_flattened = jnp.concatenate(
        [
            all_map.reshape(all_map.shape[0], -1),
            teammate_dashboard,
            teammate_directions,
            inventory,
            potions,
            intrinsics,
            direction,
            armour,
            armour_enchantments,
            special_values_per_player,
            special_values_level[None, :].repeat(static_params.player_count, axis=0),
        ],
        axis=1,
    )

    return all_flattened

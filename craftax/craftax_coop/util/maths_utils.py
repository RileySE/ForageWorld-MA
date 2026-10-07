import jax
import jax.numpy as jnp

# For utility functions - functions called more than once in meaningfully different parts of the codebase
# With the additional constraint that the functions make no reference (i.e. don't import from) any Craftax code


def random_choice(key, a, p, shape=()):
    """Same result as jax.random.choice(key, a, shape, replace=True, p=p), without its search loop.

    jax.random.choice draws via jnp.searchsorted, whose default method is a binary search of
    ceil(log2(len(a) + 1)) sequential loop steps; inside the per-mob scans and world generation
    those loops dominated the run time. This repeats jax's arithmetic but searches with
    method="compare_all" (one vectorized comparison using the same comparator), so it returns the
    same element for the same key. As in jax.random.choice, an integer `a` means arange(a).
    """
    a = jnp.asarray(a)
    p = jnp.asarray(p)
    if not jnp.issubdtype(p.dtype, jnp.inexact):
        p = p.astype(jnp.result_type(float))
    p_cuml = jnp.cumsum(p)
    r = p_cuml[-1] * (1 - jax.random.uniform(key, shape, dtype=p_cuml.dtype))
    index = jnp.searchsorted(p_cuml, r, method="compare_all").astype(int)
    return index if a.ndim == 0 else jnp.take(a, index, axis=0)


def get_distance_map(position, map_size):
    dist_x = jnp.abs(jnp.arange(0, map_size[0]) - position[0])
    dist_x = jnp.expand_dims(dist_x, axis=1)
    dist_x = jnp.tile(dist_x, (1, map_size[1]))

    dist_y = jnp.abs(jnp.arange(0, map_size[1]) - position[1])
    dist_y = jnp.expand_dims(dist_y, axis=0)
    dist_y = jnp.tile(dist_y, (map_size[0], 1))

    coords = jnp.stack([dist_x, dist_y], axis=-1)

    def _euclid_distance(x):
        return jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    dist = jax.vmap(jax.vmap(_euclid_distance))(coords)

    return dist


def get_all_players_distance_map(position, mask, static_params):
    player_proximity_map = jax.vmap(get_distance_map, in_axes=(0, None))(
        position, static_params.map_size
    )
    max_dist = jnp.sqrt(static_params.map_size[0]**2 + static_params.map_size[1]**2)
    
    # If player is dead, remove from distance consideration
    player_proximity_map_masked = jnp.where(
        mask[:, None, None],
        player_proximity_map,
        jnp.full((static_params.player_count, static_params.map_size[0], static_params.map_size[1]), max_dist)
    )
    
    all_players_proximity_map = jnp.min(player_proximity_map_masked, axis=0).astype(jnp.float32)
    return all_players_proximity_map

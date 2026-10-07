"""The faster mob code must match the code it replaced, bit for bit."""
import jax
import jax.numpy as jnp
import numpy as np

from craftax_coop.constants import DIRECTIONS_PASSIVE
from craftax_coop.util.maths_utils import random_choice


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

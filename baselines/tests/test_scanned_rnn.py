import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from seperate_ippo_rnn import ScannedRNN


class _ReferenceScannedRNN(nn.Module):
    """The original ScannedRNN: nn.GRUCell under nn.scan."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def test_scanned_rnn_matches_gru_cell():
    num_steps, batch, in_features, hidden = 64, 16, 128, 512
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    ins = jax.random.normal(keys[0], (num_steps, batch, in_features))
    resets = jax.random.uniform(keys[1], (num_steps, batch)) < 0.05
    carry = jax.random.normal(keys[2], (batch, hidden))

    reference = _ReferenceScannedRNN()
    params = reference.init(keys[3], carry, (ins, resets))
    # Same parameter names and initial values.
    new_params = ScannedRNN().init(keys[3], carry, (ins, resets))
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(params)
    for a, b in zip(jax.tree_util.tree_leaves(new_params), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # Same function, up to float rounding.
    with jax.default_matmul_precision("float32"):
        expected_carry, expected_ys = reference.apply(params, carry, (ins, resets))
        actual_carry, actual_ys = ScannedRNN().apply(params, carry, (ins, resets))
    np.testing.assert_allclose(np.asarray(actual_ys), np.asarray(expected_ys), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_carry), np.asarray(expected_carry), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(ScannedRNN.initialize_carry(batch, hidden)),
        np.asarray(_ReferenceScannedRNN.initialize_carry(batch, hidden)),
    )

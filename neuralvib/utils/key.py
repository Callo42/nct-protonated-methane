"""Jax random key utilities"""

import jax
import jax.numpy as jnp


def key_batch_split(key: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array]:
    """Like jax.random.split, but returns one subkey per batch element.

    Args:
        key: jax.PRNGkey
        batch_size: the batch size.

    Returns:
        key: jax.PRNGkey, shape (2,)
        batch_keys: jax.PRNGkeys, shape (batch_size, 2)
    """
    key, *batch_keys = jax.random.split(key, num=batch_size + 1)
    return key, jnp.asarray(batch_keys)

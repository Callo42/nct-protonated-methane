"""Geometry helper utilities for CH4."""

import jax
import jax.numpy as jnp


def _calculate_angle(vec_1: jax.Array, vec_2: jax.Array) -> jax.Array:
    """Return the angle between two vectors.

    Args:
        vec_1: (dim,) coordinates of first vector.
        vec_2: (dim,) coordinates of second vector.
    """
    vec_1_norm = jnp.linalg.norm(vec_1)
    vec_2_norm = jnp.linalg.norm(vec_2)
    vec_1_dot_vec_2 = jnp.dot(vec_1, vec_2)
    alpha = jnp.arccos(vec_1_dot_vec_2 / (vec_1_norm * vec_2_norm))
    return alpha

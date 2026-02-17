"""Normal <-> Cartesian displacement transforms for CH4."""

import jax
import jax.numpy as jnp


def _normal_to_cartesian_displacement(
    pes: callable,
    x_e: jax.Array,
    expected_eigval: jax.Array,
) -> jax.Array:
    """Calculate transformation U such that x = U Q (redundant normals -> mwcd).

    Args:
        pes: PES that accepts mass-weighted Cartesian displacement and returns energy.
        x_e: (deg_of_freedom,) coordinates at equilibrium (typically zeros).
        expected_eigval: Expected eigenvalues of Hessian for internal consistency.
    """
    hessian = jax.hessian(pes)(x_e)
    eigval, eigvectors = jnp.linalg.eigh(hessian)
    if not jnp.allclose(eigval, expected_eigval):
        raise ValueError(
            "Unexpected Hessian spectrum for CH4; re-check arg_sort_index mapping."
        )
    return eigvectors

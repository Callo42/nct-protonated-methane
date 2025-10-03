"""Transform McCoy's NN PES to jax"""

import pickle
import os
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as flax_nn

jax.config.update("jax_enable_x64", True)
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def convert_cartesian_to_sorted_cm(
    cartesian_coors: jax.Array,
) -> jax.Array:
    """Convert cartesian coordinates to CM descriptor.
    For CM descriptor, see J. Phys. Chem. A 2021, 125, 5849-5859,
    J. Phys. Chem. A 2022, 126, 4013-4024 and
    Advances in Neural Information Processing Systems, 2012; pp 440-
    448

    Args:
        cartesian_coors: (deg_of_freedom,) the FLATTENED cartesian
            coordinates in a.u.

    Returns:
        sorted_cm: (num_of_atoms,num_of_atoms) the
            SORTED CM_{ij}
            matrix described in equation 3 of
            J. Phys. Chem. A 2021, 125, 5849-5859.
            CM_{ij} is sorted s.t.
                ||C_i|| >= ||C_{i+1}|| for any i
                where C_i denotes the ith row of the CM
            see
            Advances in Neural Information Processing Systems, 2012; pp 440-
            448 for more details.
    """
    num_of_atoms = 6
    spatial_dim = 3
    nuclear_charges = [6, 1, 1, 1, 1, 1]
    cartesian_coors_flatten = cartesian_coors.reshape(-1)
    cartesian_reshaped = cartesian_coors_flatten.reshape(num_of_atoms, spatial_dim)
    init_cm = jnp.zeros((num_of_atoms, num_of_atoms), dtype=jnp.float64)
    for i in range(6):
        for j in range(6):
            if i == j:
                init_cm = init_cm.at[i, j].set(0.5 * nuclear_charges[i] ** (2.4))
            else:
                rij = jnp.linalg.norm(cartesian_reshaped[i] - cartesian_reshaped[j])
                init_cm = init_cm.at[i, j].set(
                    nuclear_charges[i] * nuclear_charges[j] / rij
                )
    reorder_index = jnp.linalg.norm(init_cm, axis=1).argsort()[::-1]
    sorted_cm = init_cm[reorder_index, :]
    sorted_cm = sorted_cm[:, reorder_index]

    return sorted_cm


def convert_bounds_to_sorted_cm(
    bounds: jax.Array,
) -> jax.Array:
    """Convert bounds to CM descriptor.
    For CM descriptor, see J. Phys. Chem. A 2021, 125, 5849-5859,
    J. Phys. Chem. A 2022, 126, 4013-4024 and
    Advances in Neural Information Processing Systems, 2012; pp 440-
    448

    NOTE: currently only implemented for CH5+!

    Args:
        bounds: (r1,r2,r3,r4,r5,r12,r13,r14,r15,r23,r24,r25,r34,r35,r45)

    Returns:
        sorted_cm: (num_of_atoms,num_of_atoms) the
            SORTED CM_{ij}
            matrix described in equation 3 of
            J. Phys. Chem. A 2021, 125, 5849-5859.
            CM_{ij} is sorted s.t.
                ||C_i|| >= ||C_{i+1}|| for any i
                where C_i denotes the ith row of the CM
            see
            Advances in Neural Information Processing Systems, 2012; pp 440-
            448 for more details.
    """
    r1, r2, r3, r4, r5, r12, r13, r14, r15, r23, r24, r25, r34, r35, r45 = bounds
    num_of_atoms = 6
    nuclear_charges = [6, 1, 1, 1, 1, 1]
    rij_matrix = np.zeros((num_of_atoms, num_of_atoms))
    rij_matrix[0, 1] = r1
    rij_matrix[0, 2] = r2
    rij_matrix[0, 3] = r3
    rij_matrix[0, 4] = r4
    rij_matrix[0, 5] = r5
    rij_matrix[1, 2] = r12
    rij_matrix[1, 3] = r13
    rij_matrix[1, 4] = r14
    rij_matrix[1, 5] = r15
    rij_matrix[2, 3] = r23
    rij_matrix[2, 4] = r24
    rij_matrix[2, 5] = r25
    rij_matrix[3, 4] = r34
    rij_matrix[3, 5] = r35
    rij_matrix[4, 5] = r45
    rij_matrix = rij_matrix + rij_matrix.T

    init_cm = jnp.zeros((num_of_atoms, num_of_atoms), dtype=jnp.float64)
    for i in range(6):
        for j in range(6):
            if i == j:
                init_cm = init_cm.at[i, j].set(0.5 * nuclear_charges[i] ** (2.4))
            else:
                rij = rij_matrix[i, j]
                init_cm = init_cm.at[i, j].set(
                    nuclear_charges[i] * nuclear_charges[j] / rij
                )
    reorder_index = jnp.linalg.norm(init_cm, axis=1).argsort()[::-1]
    sorted_cm = init_cm[reorder_index, :]
    sorted_cm = sorted_cm[:, reorder_index]

    return sorted_cm


def get_nn_pes_input(
    sorted_cm: jax.Array,
) -> jax.Array:
    """Get the direct input to NN PES
    which would be the upper trianglar elements
    excluding diagonal elements of the sorted cm, FLATTENED.

    Args:
        sorted_cm: (num_of_atoms,num_of_atoms) the
            SORTED CM_{ij}
            matrix described in equation 3 of
            J. Phys. Chem. A 2021, 125, 5849-5859.
            CM_{ij} is sorted s.t.
                ||C_i|| >= ||C_{i+1}|| for any i
                where C_i denotes the ith row of the CM
            see
            Advances in Neural Information Processing Systems, 2012; pp 440-
            448 for more details.

    Returns:
        input_to_nn: (15,) the upper triangle excluding
            the diagonal elements of the sorted CM matrix.
            NOTE: here input_to_nn is reshaped to have one leading
            dimension to compatible with pes api.
    """
    triu_indices = jnp.triu_indices_from(sorted_cm, k=1)
    input_to_nn = sorted_cm[triu_indices]
    return input_to_nn


class McCoyNNPES(flax_nn.Module):
    """The NN-PES of McCoy's
    implemented in flax.

    NOTE: the resultant energy value
        is in cm-1
    """

    out_dims: int

    @flax_nn.compact
    def __call__(self, x) -> jax.Array:
        x = x.reshape(-1)
        x = flax_nn.Dense(120)(x)
        x = flax_nn.swish(x)
        x = flax_nn.Dense(120)(x)
        x = flax_nn.swish(x)
        x = flax_nn.Dense(120)(x)
        x = flax_nn.swish(x)
        x = flax_nn.Dense(self.out_dims)(x)
        x = jnp.exp(x)
        x = x - 100
        return x


def trans_tf_params_to_flax(
    model_tf,
) -> dict:
    """Transform the tf model parameters
    to the parameters used in flas.

    NOTE: this function is specified for
    McCoy's PES ONLY!

    Args:
        model_tf: the original tensorflow model
            of McCoy's PES.

    Returns:
        params_flax: the dict of flax model parameters.
    """
    tf_weights = model_tf.get_weights()
    d0w, d0b, d1w, d1b, d2w, d2b, d3w, d3b = tf_weights
    params_flax = {
        "params": {
            "Dense_0": {
                "kernel": jnp.array(d0w, dtype=jnp.float64),
                "bias": jnp.array(d0b, dtype=jnp.float64),
            },
            "Dense_1": {
                "kernel": jnp.array(d1w, dtype=jnp.float64),
                "bias": jnp.array(d1b, dtype=jnp.float64),
            },
            "Dense_2": {
                "kernel": jnp.array(d2w, dtype=jnp.float64),
                "bias": jnp.array(d2b, dtype=jnp.float64),
            },
            "Dense_3": {
                "kernel": jnp.array(d3w, dtype=jnp.float64),
                "bias": jnp.array(d3b, dtype=jnp.float64),
            },
        }
    }
    return params_flax


def save_flax_params(params_flax: dict) -> None:
    """save flax params to file

    Args:
        params_flax: the dict of flax model parameters.
    """
    filename = "params_flax"
    with open(filename, "wb") as f:
        pickle.dump(params_flax, f)


def load_flax_params(filename: str) -> dict:
    """Load flax params to dict

    Args:
        filename: the params_flax file
    """
    with open(filename, "rb") as f:
        params_flax = pickle.load(f)
    return params_flax

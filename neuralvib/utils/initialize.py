"""Initialize Relevant Variables"""

from itertools import permutations

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.ch5_plus_jacobi import config2jacobi
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
    equilibrium_mccoy_jpca_2021_125_5849_5859,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.mcmc import calculate_center_mass_coor, rotate_to_eckart_frame
from neuralvib.utils.mcmc import _align_to_axes


def init_batched_x(
    key: jax.Array,
    batch_size: int,
    num_orb: int,
    num_of_particles: int,
    dim: int,
    init_width: float = 1.00,
    molecule: str | None = None,
    **kwargs,
) -> jax.Array:
    """Initialize batched particle configurations.

    This function initializes a batch of particle configurations with a specified
    number of particles and spatial dimensions. If no molecule is specified, the
    positions are initialized using a Gaussian distribution. For specific molecules
    like "CH5+", special initialization procedures are followed.

    Args:
        key: the jax PRNG key used for random number generation.
        batch_size: the number of configurations to generate.
        num_orb: the number of orbitals (total number of states)
        num_of_particles: the total number of particles in each configuration.
        dim: the spatial dimensionality of the system.
        init_width: the standard deviation of the Gaussian distribution used for
                    initialization when no molecule is specified.
        molecule: optional; specifies the molecule type for special initialization.
                  Currently supports "CH5+".

    Returns:
        init_x: a JAX array of shape
            (batch_size, num_orb,num_of_particles, dim) representing
            the initialized configurations.
    """
    particles = kwargs.get("particles", None)
    masses = kwargs.get("particle_mass", None)
    if particles is not None and masses is not None:
        ms = []
        for particle in particles:
            ms.append([masses[particle]] * 3)
        ms = np.array(ms)
    output_shape = (batch_size, num_orb, num_of_particles, dim)
    if molecule is None:
        init_x = init_width * jax.random.normal(
            key,
            shape=output_shape,
        )
    elif molecule == "CH5+Jacobi":
        print(f"Initializing x for {molecule}")
        init_x = init_width * jax.random.normal(
            key,
            shape=output_shape,
        )
        # single_x = saddle_c2v_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        # single_x = config2jacobi(single_x)
        # init_x = jnp.array([single_x] * (batch_size * num_orb)).reshape(
        #     batch_size, num_orb, num_of_particles, dim
        # )
    elif molecule == "CH5+":
        hydrogens = init_width * jax.random.normal(
            key, shape=(batch_size * num_orb, num_of_particles - 1, dim)
        )
        carbons = jnp.zeros(shape=(batch_size * num_orb, 1, dim))
        init_x = jnp.concatenate((carbons, hydrogens), axis=1)
        init_x = init_x.reshape(output_shape)
    elif molecule == "CH5+NoCarbon":
        print("Initializing x for CH5+ NoCarbon")
        single_x = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)[1:]
        init_x = jnp.array([single_x] * (batch_size * num_orb)).reshape(
            batch_size, num_orb, num_of_particles, dim
        )
    else:
        raise ValueError(f"Unsupported molecule type: {molecule}")

    return init_x

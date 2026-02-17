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
from neuralvib.utils.frame import fix_eckart_frame
from neuralvib.utils.mcmc import calculate_center_mass_coor, rotate_to_eckart_frame
from neuralvib.utils.mcmc import _align_to_axes


def _init_batched_x(
    key: jax.Array,
    batch_size: int,
    num_orb: int,
    num_of_particles: int,
    dim: int,
    init_width: float = 1.00,
    molecule: str | None = None,
    ms: np.ndarray | None = None,
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
        ms: (num_of_particles, dim) array of masses for each particle.

    Returns:
        init_x: a JAX array of shape
            (batch_size, num_orb, num_of_particles, dim) representing
            the initialized configurations.
    """
    output_shape = (batch_size, num_orb, num_of_particles, dim)

    if molecule in ("CH5+", "CH4"):
        print(f"Direct Gaussian Initialization in CoM-frame for {molecule}")
        if ms is None:
            raise ValueError(
                "Mass array `ms` must be provided for CoM initialization."
            )
        print(f"Initializing x for {molecule} with Gaussian CoM-frame coordinates")

        # 1) Sample Gaussian configs
        flat_shape = (batch_size * num_orb, num_of_particles, dim)
        raw_x = init_width * jax.random.normal(
            key,
            shape=flat_shape,
        )

        ms_arr = jnp.array(ms)

        # 2) Shift each configuration into its center-of-mass frame
        def _to_com(x_single: jax.Array) -> jax.Array:
            com = calculate_center_mass_coor(ms_arr, x_single)
            return x_single - com

        raw_x_com = jax.vmap(_to_com)(raw_x)

        # 3) Reshape back to (batch_size, num_orb, num_particles, dim)
        init_x = raw_x_com.reshape(output_shape)
    else:
        raise ValueError(f"Unsupported molecule type: {molecule}")

    return init_x


def init_batched_x(
    key: jax.Array,
    batch_size: int,
    num_orb: int,
    num_of_particles: int,
    dim: int,
    init_width: float = 1.00,
    init_ref_noise: float = 0.0,
    molecule: str | None = None,
    ms: np.ndarray | None = None,
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
        init_ref_noise: standard deviation of Gaussian noise added to `x_ref`
            before fixing to the Eckart frame (only used when `x_ref` is
            provided / reference-geometry initialization).
        molecule: optional; specifies the molecule type for special initialization.
                  Currently supports "CH5+".
        ms: (num_of_particles, dim) array of masses for each particle.

    Returns:
        init_x: a JAX array of shape
            (batch_size, num_orb, num_of_particles, dim) representing
            the initialized configurations.
    """
    output_shape = (batch_size, num_orb, num_of_particles, dim)
    print(f"Init ref noise={init_ref_noise}")

    if molecule in ("CH5+", "CH4"):
        print("Initialization using reference geometry for CH5+ or CH4")
        x_ref = kwargs.get("x_ref", None)
        if x_ref is None:
            raise ValueError(
                "`x_ref` must be provided for reference-geometry initialization."
            )
        if ms is None:
            raise ValueError("Mass array `ms` must be provided for frame-fixed initialization.")
        x_ref_arr = jnp.array(x_ref)
        if x_ref_arr.shape == (num_of_particles * dim,):
            x_ref_arr = x_ref_arr.reshape((num_of_particles, dim))
        if x_ref_arr.shape == (num_of_particles, dim):
            # Add a small noise around the equilibrium geometry before fixing to the
            # Eckart frame. This avoids starting all walkers at exactly the same
            # point while keeping a consistent gauge (translation + rotation fixed).
            if init_ref_noise < 0.0:
                raise ValueError(
                    f"`init_ref_noise` must be non-negative; got {init_ref_noise}."
                )

            if init_ref_noise == 0.0:
                # Keep a consistent coordinate convention across the codebase by
                # applying the same frame-fix used in the wavefunction ansatz.
                x_ref_fixed = fix_eckart_frame(x_ref_arr, ms=ms, x_ref=x_ref_arr)
                base = jnp.broadcast_to(x_ref_fixed, (num_orb, num_of_particles, dim))
                init_x = jnp.broadcast_to(base, output_shape)
                return init_x

            noise = init_ref_noise * jax.random.normal(
                key,
                shape=output_shape,
                dtype=x_ref_arr.dtype,
            )
            x0 = x_ref_arr[None, None, :, :] + noise  # (B, O, N, 3)
            x0_flat = x0.reshape((batch_size * num_orb, num_of_particles, dim))
            x0_fixed = fix_eckart_frame(x0_flat, ms=ms, x_ref=x_ref_arr)
            init_x = x0_fixed.reshape(output_shape)
        else:
            raise ValueError(
                "`x_ref` must have shape (num_particles, dim) or "
                f"({num_of_particles * dim},); got {x_ref_arr.shape}."
            )
    else:
        raise ValueError(f"Unsupported molecule type: {molecule}")

    return init_x

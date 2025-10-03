"""Energy Estimator"""

import copy
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from neuralvib.utils.parallel import joblib_parallel


class EnergyEstimator:
    """Energy Estimator

    Attributes:
        self.wf_ansatz: the callable wave function ansatz
        self.potential_func: the potential energy function
        self.compute_kinetic_forloop: if true, use foriloop
            to compute laplacians in kinetic estimator.
        self.pot_batch_method: the string indicating which method to use
            when defining batched_local_energy.
    """

    def __init__(
        self,
        wf_ansatz: Callable,
        potential_func: Callable,
        particles: tuple,
        particle_mass: dict,
        pot_batch_method: str = "jax.vmap",
        seperate_energy: bool = True,
        no_kinetic: bool = False,
    ) -> None:
        """Init

        Args:
            wf_ansatz: the wavefunction ansatz function
                signature: (params, x)
                returns: log_amplitude(real)
            potential_func: the potential energy function
                signature:
                    cartesian_coors: (dim * num_of_atoms,) the flattened
                        mass weight cartesian displacement coordinates,
                        in a specific order. For example, for CH4:
                        (
                            C_x C_y C_z
                            H_1x H_1y H_1z
                            H_2x H_2y H_2z
                            H_3x H_3y H_3z
                            H_4x H_4y H_4z
                        )
                        NOTE: in atomic unit!
                returns:
                    potential_energy: (1,) corresponding potential energy in Hartree.
            particles: the tuple with atoms names, for example,
                ("C","H","H","H","H","H")
            particle_mass: particle mass in a.u.
                key: atom name, value: mass in a.u.
            pot_batch_method: the string indicating which method to use
                when defining batched_local_energy.
                `jax.vmap` - use jax.vmap to parallel potential function
                `joblib` - use joblib.parallel to parallel potential function.
            seperate_energy: if True then return Kinetic and Potential
                seperately.
        """
        self.wf_ansatz: Callable = wf_ansatz
        self.local_potential_energy: Callable = potential_func
        self.particles = particles
        self.particle_mass = particle_mass
        if no_kinetic:
            print(
                "EnergyEstimator: Manually setting K=0."
                f"\nSince get no_kinetic={no_kinetic}."
            )
        self.no_kinetic = no_kinetic

        assert seperate_energy
        if pot_batch_method == "jax.vmap":
            if seperate_energy:
                self.batched_local_energy = self.vmap_batched_local_energies
            else:
                self.batched_local_energy = self.vmap_batched_local_energy
        elif pot_batch_method == "joblib":
            if seperate_energy:
                self.batched_local_energy = self.joblib_batched_local_energies
            else:
                self.batched_local_energy = self.joblib_batched_local_energy

    def local_kinetic_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        x: jax.Array,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: per batch implementation.

        Args:
            params: the flow parameter
            x: (num_of_particles,dim,) the configuration cartesian
                coordinate of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
        Returns:
            local_kinetic: (1,) the local kinetic energy.
        """
        num_of_particles, dim = x.shape
        flatten_masses = []
        for particle in self.particles:
            # (num_of_particles*dim,)
            flatten_masses.extend([self.particle_mass[particle]] * dim)
        flatten_masses = np.array(flatten_masses)
        if self.no_kinetic:
            return jnp.array(0.0, dtype=jnp.float64)

        def _wf_ansatz_flatten(x_flatten):
            """The flatten wf_ansatz
            Args:
                x_flatten: (num_of_particles*dim,) the flatten
                    coordinates
            Returns:
                log_psi: float, the wavefunction in log domain
            """
            return self.wf_ansatz(
                params, x_flatten.reshape(num_of_particles, dim), excitation_number
            )

        x_flatten = x.reshape(-1)

        # (num_of_particles*dim,)
        grad_flatten = jax.jacrev(_wf_ansatz_flatten)(x_flatten)
        laplacians = jax.hessian(_wf_ansatz_flatten)(x_flatten).diagonal()
        local_kinetic = -0.5 * jnp.sum((laplacians + grad_flatten**2) / flatten_masses)

        # local_kinetic = jnp.where(local_kinetic<0,1e4,local_kinetic)

        return local_kinetic

    def local_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        x: jax.Array,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """Local Energy Estimator

        Args:
            params: the flow parameter
            x: (num_of_particles,dim,) the configuration cartesian
                coordinate of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            local_energy: (1,) the local energy.
        """
        x_for_pes = x
        kin_energy = self.local_kinetic_energy(params, x, excitation_number)
        pot_energy = self.local_potential_energy(x_for_pes)
        local_energy = kin_energy + pot_energy
        return local_energy

    def local_energies(
        self,
        params: jax.Array | np.ndarray | dict,
        x: jax.Array,
        excitation_number: np.ndarray,
    ) -> tuple[jax.Array, jax.Array | float]:
        """Local Energy Estimator
        Return Kinetic and potential seperately

        Args:
            params: the flow parameter
            x: (num_of_particles,dim,) the configuration cartesian
                coordinate of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
        Returns:
            kin_energy: (1,) the kinetic energy
            pot_energy: (1,) the potential energy
        """
        x_for_pes = x
        kin_energy = self.local_kinetic_energy(params, x, excitation_number)
        pot_energy = self.local_potential_energy(x_for_pes)
        return (kin_energy, pot_energy)

    def joblib_batched_local_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        batched_x: jax.Array,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """Batched Local Energy Estimator

        Args:
            params: the flow parameter
            batched_x: (batch_size,num_of_particles,dim,) the batched
                configuration cartesian
                coordinate of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            batched_local_energy: the batched local energy.
        """
        batched_x_for_pes = np.array(batched_x)
        batched_kin_energy_func = jax.jit(
            jax.vmap(self.local_kinetic_energy, in_axes=(None, 0, None))
        )
        batched_kin_energy = batched_kin_energy_func(
            params, batched_x, excitation_number
        )
        batched_pot_func = joblib_parallel(self.local_potential_energy)
        batched_pot_energy = np.array(batched_pot_func(batched_x_for_pes))
        batched_local_energy = batched_kin_energy + batched_pot_energy
        return batched_local_energy

    def joblib_batched_local_energies(
        self,
        params: jax.Array | np.ndarray | dict,
        batched_x: jax.Array,
        excitation_numbers: np.ndarray,
    ) -> tuple[jax.Array, np.ndarray]:
        """Batched Local Energy Estimator
        Return Kinetic and potential seperately

        Args:
            params: the flow parameter
            batched_x: (batch_size,num_orb,num_of_particles,dim,) the batched
                configuration cartesian
                coordinate of the particle(s).
            excitation_numbers: (num_orb,num_particles*dim,)
                the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            batched_kinetics:(batch_size,num_orb) the batched kinetic energy.
            batch_potentials:(batch_size,num_orb) the batched potential energy.
        """
        batchsize, num_orb = batched_x.shape[:2]
        batched_x_copy = copy.deepcopy(batched_x)
        batched_x_for_pes = np.array(batched_x_copy)
        batched_x_for_pes = batched_x_for_pes.reshape(
            batchsize * num_orb, batched_x.shape[2], batched_x.shape[3]
        )

        batched_kin_energy_func = jax.vmap(
            jax.vmap(self.local_kinetic_energy, in_axes=(None, 0, 0)),
            in_axes=(None, 0, None),
        )
        batched_kin_energy_func = jax.jit(batched_kin_energy_func)
        batched_kin_energy = batched_kin_energy_func(
            params, batched_x, excitation_numbers
        )
        batched_kinetics = batched_kin_energy

        batched_pot_func = joblib_parallel(self.local_potential_energy)
        batched_pot_energy = np.array(batched_pot_func(batched_x_for_pes))
        batched_pot_energy = batched_pot_energy.reshape(batchsize, num_orb)
        batched_potentials = batched_pot_energy
        return batched_kinetics, batched_potentials

    def vmap_batched_local_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        batched_x: jax.Array,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """Batched Local Energy Estimator

        Args:
            params: the flow parameter
            batched_x: (batch_size,num_of_particles,dim,) the batched
                configuration cartesian
                coordinate of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            batched_local_energy: the batched local energy.
        """
        batched_local_energy = jax.vmap(self.local_energy, in_axes=(None, 0, None))(
            params, batched_x, excitation_number
        )
        return batched_local_energy

    def vmap_batched_local_energies(
        self,
        params: jax.Array | np.ndarray | dict,
        batched_x: jax.Array,
        excitation_numbers: np.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Batched Local Energy Estimator
        Return Kinetic and potential seperately

        Args:
            params: the flow parameter
            batched_x: (batch_size,num_orb,num_of_particles,dim,) the batched
                configuration cartesian
                coordinate of the particle(s).
            excitation_numbers: (num_orb,num_particles*dim,)
                the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
        Returns:
            batched_kinetics:(batch_size,num_orb) the batched kinetic energy.
            batch_potentials:(batch_size,num_orb) the batched potential energy.
        """
        batched_local_energies = jax.vmap(
            jax.vmap(self.local_energies, in_axes=(None, 0, 0)), in_axes=(None, 0, None)
        )(params, batched_x, excitation_numbers)
        batched_local_energies = jnp.array(batched_local_energies)
        batched_kinetics = batched_local_energies[0]
        batched_potentials = batched_local_energies[1]
        return batched_kinetics, batched_potentials

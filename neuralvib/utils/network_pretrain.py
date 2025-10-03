"""Pretrain the network"""

from functools import partial
import time
import os
import warnings
from typing import Callable
import copy
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from optax._src import base as optax_base

from neuralvib.molecule.ch5plus.ch5_plus_jacobi import CH5PlusJacobi, config2jacobi
from neuralvib.utils.update import clip_grad_norm
from neuralvib.wfbasis.basis import log_gaussian_1d
from neuralvib.wfbasis.basis import hermite
from neuralvib.molecule.utils.init_molecule import InitMolecule
from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.molecule.ch5plus.ch5_plus import CH5Plus
from neuralvib.utils.convert import _convert_Hartree_to_inverse_cm

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#222222"
plt.rcParams["axes.facecolor"] = "#222222"


def _log_wf_base_1d(x: float, x0: float, m: float, w: float, n: int) -> jax.Array:
    """The wave function of 1D Harmonic Oscillator.
    NOTE: centered at x0
    psi_n(x)
    = (1/sqrt(2^n*n!)) * (m*w/(hbar*pi))^(1/4)
        * e^(-m*w*(x-x0)^2/(2hbar)) * Hn(sqrt(m*w/hbar)(x-x0))

    NOTE: the `hermite` here is with coefficients like
        (1/sqrt(2^n*n!)) * (1/pi)^(1/4)
        hence here log_psi doesn't contain those n-relative
        coefficients.
    NOTE: 1D! The eigenstates of a one-dimensional
    harmonic oscillator with frequency w, centered at x=0.

    Args:
        x: the 1D coordinate of the (single) coordinate.
        x0: the center of the oscillator
        m: the mass of the particle in a.u.
        w: the frequency of the oscillator in a.u.
        n: the excitation quantum number

        NOTE: n=0 for GS!

    Returns:
        log_psi: float, the log probability amplitude at x.
        NOTE: this is log|psi|
    """

    log_psi = (
        jnp.log(m * w) / 4
        - 0.5 * m * w * (x - x0) ** 2
        + jnp.log(jnp.abs(hermite(n, jnp.sqrt(m * w) * (x - x0))))
    )
    return log_psi


class HermiteFunctionAtX0:
    """Harmonic Oscillator Wavefunction
    (Hermite Functions, which is often called Hermite-Gaussian functions)
    with permutative invariance
    NOTE: this is a Hermite Function centered at specific x0
    rather than centered at 0!
    """

    def __init__(
        self,
        particles: tuple,
        m: dict,
        w: dict,
        x0: np.ndarray,
    ) -> None:
        """Init
        Args:
            particles: the tuple with atoms names, for example,
                ("C","H","H","H","H","H")
            m: the mass of the particle in a.u.
                with key as the atom name and value as the mass
            w: the frequency of each particle in a.u.
                with key as the atom name and value as the frequency
            x0: (num_particles,dim,) the centers of each atom corresponding
                to the `particles`
        """
        self.particles = particles
        self.x0 = x0
        num_particles, dim = x0.shape
        ms = []
        ws = []
        for particle in particles:
            ms.extend([m[particle]] * dim)
            ws.extend([w[particle]] * dim)
        self.ms = np.array(ms)
        self.ws = np.array(ws)
        assert len(self.x0) == len(self.particles)
        assert x0.shape[1] == 3

    def log_hermite_func(
        self,
        coors: jax.Array | np.ndarray,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """the permutational invariant hermite functions
        NOTE: Centered at x0
        Args:
            coors: (num_particles,dim) the full configuration coordinates
                of the system, ordered as in __init__, `particles`.
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
        Returns:
            phi_base: the full invariant hermite function wavefunction
                phi_base = log_hermite1 + log_hermite2 + ...
        """
        coors = coors.reshape(-1)
        x0 = self.x0.reshape(-1)
        phis = jax.vmap(_log_wf_base_1d, in_axes=(0, 0, 0, 0, 0))(
            coors, x0, self.ms, self.ws, excitation_number
        )
        phi_base = jnp.sum(phis)
        return phi_base


class SampleHermiteFunc:
    """Sample the Hermite function
    centered at specific x0
    with directly sampling gaussians
    and then change of variables scheme.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        particles: tuple,
        m: dict,
        w: dict,
        x0: np.ndarray,
        batch: int,
    ) -> None:
        self.rng = rng
        self.particles = particles
        self.m = m
        self.w = w
        self.x0 = x0  # (num_particles,dim,) the centers of each atom corresponding
        # to the `particles`
        self.batch = batch
        assert len(self.x0) == len(self.particles)
        assert x0.shape[1] == 3

    def sampler(self) -> np.ndarray:
        """Sample the Hermite function.
        Returns:
            x: (batch, num_particles, dim)
        """
        ms = []
        ws = []
        for particle in self.particles:
            ms.extend([self.m[particle]] * 3)
            ws.extend([self.w[particle]] * 3)
        ms = np.array(ms)
        ws = np.array(ws)
        locs = self.x0.reshape(-1)
        scale = np.ones_like(locs) / np.sqrt(ms * ws)
        # x = self.rng.normal(loc=locs, scale=scale, size=(self.batch, len(locs)))
        x = np.random.normal(loc=locs, scale=scale, size=(self.batch, len(locs)))
        x = x.reshape(self.batch, len(self.particles), 3)
        return x


class FlowPretrain:
    """Pretrain the Flow to fit the harmonic
    approximation wavefunction
    of specific molecule.

    NOTE: Flow is pretrained according to the
    GS of the normal modes, hence currently
    only accept excitation number = GS for pretrain.

    NOTE: this is a per molecule implementation!
    Since the equilibrium configuration of different
    molecule would be different, as well as the wave function.

    NOTE: Please remember to release the memory that
    has been allocated to FlowPretrain instance
    after pretrain!
    """

    def __init__(
        self,
        molecule_init_obj: InitMolecule,
        key: jax.Array,
        log_wf_ansatze: Callable[[optax_base.Params | dict, jax.Array], jax.Array],
        excitation_number: np.ndarray,
        init_params: optax_base.Params | dict,
        iterations: int,
        pretrain_batch: int,
        data_path: str | None = None,
        tolerance: float = 1e-3,
    ) -> None:
        """Init the object
        Args:
            molecule_init_obj: the molecule obj representing to which molecule the flow
                would be pretrained s.t. the wavefunciton is close to
                the harmonic approximation of the molecule.
            key: the PRNGKey.
            log_wf_ansatze: the wavefunction ansatz function
                signature: (params, x)
                returns: log_amplitude(real)
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
                 NOTE: currently only accept excitation_number = GS
            init_params: the init network parameters.
            iterations: the attempt iterations for pretrain. If after
                iterations times of pretrain and tolerance is not reached,
                raise error.
            pretrain_batch: the batch size for pretrain.
            data_path: the data directory path. Only needed in plot.
        """
        self.excitation_number = excitation_number
        # self.permute_hydrogens = True
        self.permute_hydrogens = False

        self.log_wf_ansatze: Callable[
            [optax_base.Params | dict, jax.Array], jax.Array
        ] = log_wf_ansatze
        self.init_params: optax_base.Params | dict = init_params
        self.iterations: int = iterations
        self.pretrain_batch: int = pretrain_batch
        self.molecule_init_obj: InitMolecule = copy.deepcopy(molecule_init_obj)
        self.data_path = data_path
        self.tolerance = tolerance

        # self.loss_scheme:str = "KL"
        # self.loss_scheme:str = "cross entropy"

        if self.molecule_init_obj.molecule == "test":
            batch = 100000
            key, subkey = jax.random.split(key)
            target_center = 3.0
            sigma = 8.0
            sigma = np.array([sigma] * 3)
            x0s = np.array([target_center] * 3)

            def _logp_fn(xs: jax.Array) -> jax.Array:
                """Vmapped logP func for computing entropy
                NOTE: for test case!
                NOTE: and this is the TARGET wf for test
                Args:
                    xs: (batch,num_of_particles,dim) here in test case
                        xs would always be (batch,1,3)
                Returns:
                    logp: (batch,) the logp for each batch
                """
                xs = xs.reshape(batch, 3)
                log_gaussian = jax.vmap(log_gaussian_1d)
                logps = jax.vmap(log_gaussian, in_axes=(0, None, None))(xs, x0s, sigma)
                return 2 * logps.sum(axis=1)

            @jax.jit
            def _draw_samples(key) -> jax.Array:
                """Sampler wrapper"""
                coors = sigma * jax.random.normal(key, (batch, 1, 3)) + target_center
                return coors

            self.sampler = _draw_samples
            print("......Draw Samples From Test WaveFunction......")
            key, subkey = jax.random.split(key)
            coors = self.sampler(subkey)
            print("Sampling w.r.t. normal coordinates done,")

            print(
                "......Computing Information Entropy of the target distribution......"
            )
            self.target_distr_infor_entropy = self._compute_target_entropy(
                samples=coors,
                logp_fn=_logp_fn,
            )
            print(f"Target distribution H(p) = {self.target_distr_infor_entropy}")

        elif self.molecule_init_obj.molecule == "CH5+":
            assert np.all(excitation_number == np.zeros(18, dtype=int))
            assert isinstance(molecule_init_obj.molenet_molecule_obj, CH5Plus)
            print(
                "***************************************************\n"
                "***Pretraining the network w.r.t"
                f" {self.molecule_init_obj.molecule}...*****\n"
                "***NOTE that this pretrain only trains the network*\n"
                "***with the ground state wavefunction and      ****\n"
                "***minimizing the cross entropy between the    ****\n"
                "***initialized GS wavefunction with the GS     ****\n"
                "***wavefunction that is represented in normal  ****\n"
                "***coordinates.                                ****\n"
                "***************************************************\n"
            )
            batch = self.pretrain_batch

            particles = self.molecule_init_obj.particles
            mass = self.molecule_init_obj.particle_mass
            omegas = self.molecule_init_obj.omega_for_pretrain

            print(
                f"mass={mass}\nomegas={omegas}\n"
                # f"mass*omega={mass*omegas}\n"
                # f"NormalizeFactor(sqrt(mass*omega/pi))={np.sqrt(mass*omegas/np.pi)}"
            )
            # if np.sqrt(mass * omegas / np.pi) > 1.0:
            #     warnings.warn(
            #         "Typically for good pretrain behaviour, currently "
            #         "requires that sqrt(mass*omega/pi) < 1.0, "
            #         f"get {np.sqrt(mass*omegas/np.pi)}"
            #     )
            time.sleep(5)

            x0 = self.molecule_init_obj.pretrain_x0
            x0 = x0.reshape(6, 3)
            print("=" * 50)
            print("Sampling")
            print(f"Target centered at \n {x0}")
            print("=" * 50)
            rng = np.random.default_rng()
            hermite_func_sampler_obj = SampleHermiteFunc(
                rng=rng,
                particles=particles,
                m=mass,
                w=omegas,
                x0=x0,
                batch=batch,
            )
            self.hermite_func_sampler_obj = hermite_func_sampler_obj
            self.sampler = hermite_func_sampler_obj.sampler
            samples = self.sampler()

            print("=" * 50)
            print("Initializing Target WF")
            print("=" * 50)
            target_wf_obj = HermiteFunctionAtX0(
                particles=particles,
                m=mass,
                w=omegas,
                x0=x0,
            )

            def _logp_fn(xs: jax.Array | np.ndarray) -> jax.Array:
                """Vmapped logP func for computing entropy
                NOTE: Target function
                Args:
                    xs: (batch,num_of_particles,dim)
                Returns:
                    logps: (batch,) the logp for each batch
                """
                logps = jax.vmap(target_wf_obj.log_hermite_func, in_axes=(0, None))(
                    xs, self.excitation_number
                )
                logps *= 2
                return logps

            if self.permute_hydrogens:
                print("=" * 50)
                print("Permuting Hydrogens")
                print("=" * 50)
                samples = self.permute_atoms(
                    samples_at_one_eq=samples,
                    molecule=self.molecule_init_obj.molecule,
                )

            # print("=" * 50)
            # print("Computing Potentials After Permute")
            # print("=" * 50)
            # _pot = self.compute_potential(samples=samples_permuted)
            # _pot = _convert_Hartree_to_inverse_cm(_pot)
            # self._pot_after_permute = _pot
            # print(f"Potential after permute = {_pot} cm-1")
            # time.sleep(5)
            # del _pot

            print(
                "......Computing Information Entropy of the target distribution......"
            )
            self.target_distr_infor_entropy = self._compute_target_entropy(
                samples=samples,
                logp_fn=_logp_fn,
            )
            print(f"Target distribution H(p) = {self.target_distr_infor_entropy}")
            del target_wf_obj
            del _logp_fn
            del samples
        elif self.molecule_init_obj.molecule == "CH5+Jacobi":
            assert np.all(excitation_number == np.zeros(15, dtype=int))
            assert isinstance(molecule_init_obj.mole_instance, CH5PlusJacobi)
            print(
                "***************************************************\n"
                "***Pretraining the network w.r.t"
                f" {self.molecule_init_obj.molecule}...*****\n"
                "***NOTE that this pretrain only trains the network*\n"
                "***with the ground state wavefunction and      ****\n"
                "***minimizing the cross entropy between the    ****\n"
                "***initialized GS wavefunction with the GS     ****\n"
                "***wavefunction that is represented in normal  ****\n"
                "***coordinates.                                ****\n"
                "***************************************************\n"
            )
            batch = self.pretrain_batch

            particles = self.molecule_init_obj.particles
            mass = self.molecule_init_obj.particle_mass
            omegas = self.molecule_init_obj.omega_for_pretrain

            print(
                f"mass={mass}\nomegas={omegas}\n"
                # f"mass*omega={mass*omegas}\n"
                # f"NormalizeFactor(sqrt(mass*omega/pi))={np.sqrt(mass*omegas/np.pi)}"
            )
            time.sleep(5)

            x0 = self.molecule_init_obj.pretrain_x0
            x0 = x0.reshape(5, 3)
            print("=" * 50)
            print("Sampling")
            print(f"Target centered at \n {x0}")
            print("=" * 50)
            rng = np.random.default_rng()
            hermite_func_sampler_obj = SampleHermiteFunc(
                rng=rng,
                particles=particles,
                m=mass,
                w=omegas,
                x0=x0,
                batch=batch,
            )
            self.hermite_func_sampler_obj = hermite_func_sampler_obj
            self.sampler = hermite_func_sampler_obj.sampler
            samples = self.sampler()

            print("=" * 50)
            print("Initializing Target WF")
            print("=" * 50)
            target_wf_obj = HermiteFunctionAtX0(
                particles=particles,
                m=mass,
                w=omegas,
                x0=x0,
            )

            def _logp_fn(xs: jax.Array | np.ndarray) -> jax.Array:
                """Vmapped logP func for computing entropy
                NOTE: Target function
                Args:
                    xs: (batch,num_of_particles,dim)
                Returns:
                    logps: (batch,) the logp for each batch
                """
                logps = jax.vmap(target_wf_obj.log_hermite_func, in_axes=(0, None))(
                    xs, self.excitation_number
                )
                logps *= 2
                return logps

            if self.permute_hydrogens:
                print("=" * 50)
                print("Permuting Hydrogens")
                print("=" * 50)
                samples = self.permute_atoms(
                    samples_at_one_eq=samples,
                    molecule=self.molecule_init_obj.molecule,
                    mole_instance=self.molecule_init_obj.mole_instance,
                )

            print(
                "......Computing Information Entropy of the target distribution......"
            )
            self.target_distr_infor_entropy = self._compute_target_entropy(
                samples=samples,
                logp_fn=_logp_fn,
            )
            print(f"Target distribution H(p) = {self.target_distr_infor_entropy}")
            del target_wf_obj
            del _logp_fn
            del samples
        else:
            raise NotImplementedError(
                f"Pretraining of {self.molecule_init_obj.molecule} not implemented!"
            )
        print(
            "--------------------------------------------\n"
            "Pretrain Configs:\n"
            f"data_path: {self.data_path}\n"
            f"batch: {batch}\n"
            # f"permute_hydrogen: {self.permute_hydrogens}\n"
        )

    def _compute_target_entropy(
        self,
        samples: jax.Array | np.ndarray,
        logp_fn: Callable[[jax.Array], jax.Array],
    ) -> float | jax.Array:
        """Compute the target distribution (information) entropy
        For comparision of entropy that the flowed wavefunction
        and the desired wavefunction.

        Args:
            samples: (batch,...) the samples that is drawn from
                the logp_fn, with leading dimension as batch
            logp_fn: callable that evaluate log-probability of batched samples.
                The signature is logp_fn(samples), where samples has shape (batch,...).
                with return shape (batch,)

        Returns:
            target_entropy: the informational entropy of target
                distribution.
        NOTE: in this case, directly computes the probability
        as squared wavefunction and then informational entropy
        equals to - int dx P(x) log P(x) = - (1/M) sum log P(xi)
        here M is the total number of samples and xi an individual
        sample that is drawn from distribution.
        """
        if self.molecule_init_obj.molecule == "test":
            logps = logp_fn(samples)
            target_entropy = -jnp.mean(logps)
        if (
            self.molecule_init_obj.molecule == "CH5+"
            or self.molecule_init_obj.molecule == "CH5+Jacobi"
        ):
            if self.permute_hydrogens:
                permuted_batchsize = len(samples)
                if permuted_batchsize % 120 != 0:
                    raise ValueError("Please check dimension!")
                batch_size = int(permuted_batchsize / 120)
                logps = logp_fn(samples)  # (batch_size*120,)
                logps = np.array(logps.reshape(batch_size, 120))
                # p(x) = sum_pi p_0(pi x)/120
                # here pi refers to permute operate
                probs = np.exp(logps)
                probs = np.sum(probs, axis=1) / 120
                logps = np.log(probs)
                target_entropy = -np.mean(logps)
            else:
                logps = logp_fn(samples)
                target_entropy = -jnp.mean(logps)
        return target_entropy

    @staticmethod
    def permute_atoms(
        samples_at_one_eq: np.ndarray,
        molecule: str,
        **kwargs,
    ) -> np.ndarray:
        """Permute the atoms
        NOTE: currently only support that all the
        atoms in samples_at_one_wq are same!

        Args:
            samples_at_one_eq: (batch,num_of_particles,dim) the samples at one
                equilibrium configuration.
            molecule: the name of the molecule, currently only support CH5+
        Returns:
            samples_permuted: (120*batch,num_of_particles,dim) the cartesians samples
                that are fully permuted.
                NOTE: return
                    #120 times enlarged samples with one of them
                    the original one and others by permuting
                    hydrogens, all in cartesians.
        """
        if molecule == "CH5+":
            permute_index = [1, 2, 3, 4, 5]
            hydrogen_all_permute = np.array(list(permutations(permute_index)))
            carbon_index = np.zeros((len(hydrogen_all_permute), 1), dtype=np.int32)
            all_permute = np.concatenate((carbon_index, hydrogen_all_permute), axis=-1)

            @jax.jit
            def permute_single_config(single_config):
                # vmap over permutations
                permuted_cartesians = jax.vmap(lambda p, x: x[p], in_axes=(0, None))(
                    all_permute, single_config
                )
                return permuted_cartesians

            # vmap over the batch of samples
            samples_permuted_batched = jax.vmap(permute_single_config)(
                samples_at_one_eq
            )
            # reshape to (batch*120, 5, 3)
            batch_size, num_perms, particles, dims = samples_permuted_batched.shape
            samples_permuted = samples_permuted_batched.reshape(
                batch_size * num_perms, particles, dims
            )

        elif molecule == "CH5+Jacobi":
            mole_instance = kwargs.get("mole_instance", None)
            jacobi2cart = jax.jit(
                lambda x: mole_instance.convert_jacobi2jbb_cartesian_input_xn(x).T
            )
            permute_index = [1, 2, 3, 4, 5]
            hydrogen_all_permute = jnp.array(list(permutations(permute_index)))
            carbon_index = jnp.zeros((len(hydrogen_all_permute), 1), dtype=np.int32)
            all_permute = jnp.concatenate((carbon_index, hydrogen_all_permute), axis=-1)

            @jax.jit
            def permute_single_config(single_config_jacobi):
                """Permutes a single jacobi configuration."""
                single_config_cart = jacobi2cart(single_config_jacobi)
                # vmap over permutations
                permuted_cartesians = jax.vmap(lambda p, x: x[p], in_axes=(0, None))(
                    all_permute, single_config_cart
                )
                # vmap over permuted cartesians
                permuted_jacobis = jax.vmap(config2jacobi)(permuted_cartesians)
                return permuted_jacobis

            # vmap over the batch of samples
            samples_permuted_batched = jax.vmap(permute_single_config)(
                samples_at_one_eq
            )
            # reshape to (batch*120, 5, 3)
            batch_size, num_perms, particles, dims = samples_permuted_batched.shape
            samples_permuted = samples_permuted_batched.reshape(
                batch_size * num_perms, particles, dims
            )
        else:
            raise NotImplementedError(f"Permute atoms for {molecule} not implemented!")
        return jnp.array(samples_permuted)

    @staticmethod
    def from_normal_to_sample(
        molecule_obj: MoleculeBase,
        normals: jax.Array,
        permute_hydrogens: bool,
    ) -> np.ndarray:
        """From normal samples generate cartesians samples

        Args:
            molecule_obj: the MoleculeBase instance of the molecule
            normals: the samples in normal coordinates
            permute_hydrogens: True for permuting hydrogens
        Returns:
            cartesians_without_carbon: the cartesians samples
                that eliminated carbon as origin.
                NOTE: if permute_hydrogens==True, then return
                    #120 times enlarged samples with one of them
                    the original one and others by permuting
                    hydrogens, all in cartesians.
        """
        batched_normals = jax.jit(
            jax.vmap(molecule_obj.convert_scaled_normal_to_normal)
        )(normals)
        # batched_cartesians: (batch,num_of_atoms,3)
        batched_cartesians = jax.jit(jax.vmap(molecule_obj.normal_to_config_cartesian))(
            batched_normals
        )
        # subtract carbon coordinate to make carbon as origin
        batched_cartesians_c_origin = []
        for single_config in batched_cartesians:
            carbon = single_config[0]
            batched_cartesians_c_origin.append(single_config - carbon)
        # batched_cartesians_c_origin: (batch,num_of_atoms,3)
        batched_cartesians_c_origin = np.array(batched_cartesians_c_origin)

        if permute_hydrogens:
            print("......Permutin Hydrogens......")
            permute_index = [1, 2, 3, 4, 5]
            hydrogen_all_permute = np.array(list(permutations(permute_index)))
            cartesians_all_permute_without_carbon = []

            for single_config in batched_cartesians_c_origin:
                if single_config.shape != (6, 3):
                    raise ValueError(
                        f"Need coordinates shape = (6,3), get {single_config.shape}"
                    )
                if (single_config[0] != np.array([0.0, 0.0, 0.0])).any():
                    raise ValueError(
                        f"Need originated at carbon. Get carbon {single_config[0]}"
                    )

                # permute
                for i, hydrogen_permute in enumerate(hydrogen_all_permute, start=1):
                    config_permutation = np.concatenate(
                        (
                            np.array([0]),
                            hydrogen_permute,
                        )
                    )
                    permuted_cartesian = single_config[config_permutation]
                    permuted_without_carbon = np.array(permuted_cartesian[1:])
                    cartesians_all_permute_without_carbon.append(
                        permuted_without_carbon
                    )
            cartesians_without_carbon = cartesians_all_permute_without_carbon
        else:
            cartesians_without_carbon = []
            for single_config in batched_cartesians_c_origin:
                if single_config.shape != (6, 3):
                    raise ValueError(
                        f"Need coordinates shape = (6,3), get {single_config.shape}"
                    )
                if (single_config[0] != np.array([0.0, 0.0, 0.0])).any():
                    raise ValueError(
                        f"Need originated at carbon. Get carbon {single_config[0]}"
                    )

                without_carbon = np.array(single_config[1:])
                cartesians_without_carbon.append(without_carbon)
        return np.array(cartesians_without_carbon)

    def compute_potential(
        self,
        samples: np.ndarray,
    ) -> float:
        """Compute potential"""
        if self.molecule_init_obj.molecule == "CH5+":
            x_for_pes = samples
            pot_fn = self.molecule_init_obj.pes_cartesian
            pot = jax.vmap(pot_fn)(x_for_pes)
            return np.mean(pot)

    def plot_train(self) -> None:
        """Plot train"""
        if self.data_path is None:
            raise ValueError(
                "When plotting training curve, need to"
                " pass data_path to FlowPretrain instance, "
                f"get {self.data_path}"
            )
        plot_path = self.data_path
        target = np.ones_like(self.epoch_list) * self.target_distr_infor_entropy
        figure_file = os.path.join(plot_path, "pretrain.png")
        fig, ax = plt.subplots(figsize=(9, 6), dpi=900)
        ax.plot(self.epoch_list, self.loss_list, label="Pretrain")
        ax.plot(self.epoch_list, target, "--", label="Target")
        ax.set_ylabel("CrossEntropy")

        ax.set_xlabel("Epoch")
        ax.set_ylim(
            [
                0,
                3 * self.target_distr_infor_entropy,
            ]
        )
        plt.title("Pretrain")

        plt.text(
            2,
            0.5,
            # f"PotAfterPremute: {self._pot_after_permute:.2f} cm-1\n"
            f"x0_for_pretrain: {self.molecule_init_obj.pretrain_x0}\n"
            f"omega_for_pretrain: {self.molecule_init_obj.omega_for_pretrain}\n",
        )
        plt.legend()
        plt.savefig(figure_file)
        print(f"Figure saved at {figure_file}")

    def pretrain(self) -> optax_base.Params:
        """Pretrain the network

        Returns:
            pretrained_params: the network parameters after pretrain.
        """
        print("*" * 30)
        print("Start Pretraining")
        print("*" * 30)
        # converge_criteria = 10
        key = jax.random.PRNGKey(42)
        init_params = copy.deepcopy(self.init_params)
        # convergence_count = 0

        if self.molecule_init_obj.molecule == "test":
            init_lr = 2e-2
        elif self.molecule_init_obj.molecule == "CH5+":
            init_lr = 1e-2
        elif self.molecule_init_obj.molecule == "CH5+Jacobi":
            init_lr = 1e-3

        optimizer = optax.adam(init_lr)
        opt_state = optimizer.init(init_params)
        params = init_params
        epoch_list = []
        loss_list = []
        # val_loss_list = []

        def _loss(params: optax_base.Params, samples: np.ndarray) -> jax.Array:
            """Loss function"""
            log_wfs = jax.vmap(self.log_wf_ansatze, in_axes=(None, 0, None))(
                params, samples, self.excitation_number
            )
            logps = 2 * log_wfs
            cross_entropy = -jnp.mean(logps)
            loss = cross_entropy
            return loss

        def _pmapped_loss(params: optax_base.Params, samples: np.ndarray) -> jax.Array:
            """Pmapped loss"""
            losses = jax.pmap(_loss, in_axes=(None, 0))(params, samples)
            loss = jnp.mean(losses)
            return loss

        if self.molecule_init_obj.molecule == "test":
            for i in range(self.iterations):
                key, subkey = jax.random.split(key, 2)
                coors = self.sampler(subkey)
                samples = jax.lax.stop_gradient(coors)
                samples = np.array(samples)
                np.random.shuffle(samples)

                def _loss_for_grad(params):
                    return _loss(params, samples)

                loss_i = _loss_for_grad(params)
                print(
                    f"Epoch: {i:06}, Loss={loss_i:.5f}",
                    f"Target={self.target_distr_infor_entropy:.5f}",
                    flush=True,
                )
                epoch_list.append(i)
                loss_list.append(loss_i)

                # if abs(loss_i - self.target_distr_infor_entropy) <= self.tolerance:
                #     convergence_count += 1
                # else:
                #     convergence_count = 0

                # if convergence_count >= converge_criteria:
                #     break

                grad = jax.grad(_loss_for_grad)(params)
                updates, opt_state = optimizer.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)

        elif self.molecule_init_obj.molecule == "CH5+":
            for i in range(self.iterations):
                samples = self.sampler()
                if self.permute_hydrogens:
                    samples = self.permute_atoms(
                        samples_at_one_eq=samples,
                        molecule="CH5+",
                    )

                num_of_device = jax.device_count()
                num_samples = len(samples)
                num_of_particles = samples.shape[1]
                dim = samples.shape[2]
                assert num_samples % num_of_device == 0
                samples = samples.reshape(
                    (num_of_device, num_samples // num_of_device, num_of_particles, dim)
                )

                loss_i = _pmapped_loss(params, samples)
                print(
                    f"Epoch: {i:06}, Loss={loss_i:.5f}",
                    f"Target={self.target_distr_infor_entropy:.5f}",
                    flush=True,
                )
                epoch_list.append(i)
                loss_list.append(loss_i)

                # if abs(loss_i - self.target_distr_infor_entropy) <= self.tolerance:
                #     convergence_count += 1
                # else:
                #     convergence_count = 0

                del loss_i

                grad = jax.grad(_pmapped_loss, argnums=0)(params, samples)
                grad = clip_grad_norm(grad, 1.0)
                updates, opt_state = optimizer.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)

                # if convergence_count >= converge_criteria:
                # break

                del grad
                del updates
        elif self.molecule_init_obj.molecule == "CH5+Jacobi":
            for i in range(self.iterations):
                samples = self.sampler()
                if self.permute_hydrogens:
                    samples = self.permute_atoms(
                        samples_at_one_eq=samples,
                        molecule="CH5+Jacobi",
                        mole_instance=self.molecule_init_obj.mole_instance,
                    )

                num_of_device = jax.device_count()
                num_samples = len(samples)
                num_of_particles = samples.shape[1]
                dim = samples.shape[2]
                assert num_samples % num_of_device == 0
                samples = samples.reshape(
                    (num_of_device, num_samples // num_of_device, num_of_particles, dim)
                )

                loss_i = _pmapped_loss(params, samples)
                print(
                    f"Epoch: {i:06}, Loss={loss_i:.5f}",
                    f"Target={self.target_distr_infor_entropy:.5f}",
                    flush=True,
                )
                epoch_list.append(i)
                loss_list.append(loss_i)

                # if abs(loss_i - self.target_distr_infor_entropy) <= self.tolerance:
                #     convergence_count += 1
                # else:
                #     convergence_count = 0

                del loss_i

                grad = jax.grad(_pmapped_loss, argnums=0)(params, samples)
                grad = clip_grad_norm(grad, 1.0)
                updates, opt_state = optimizer.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)

                # if convergence_count >= converge_criteria:
                # break

                del grad
                del updates
        else:
            raise NotImplementedError(
                f"Pretraining of {self.molecule_init_obj.molecule} not implemented!"
            )

        # if convergence_count < converge_criteria:
        #     raise TimeoutError(
        #         f"Convergence failed to reach with tolerance={self.tolerance}"
        #         f" within iterations={self.iterations} by hitting tolerance "
        #         f"{converge_criteria} times contineously."
        #     )
        self.epoch_list = np.array(epoch_list)
        self.loss_list = np.array(loss_list)
        # self.val_loss_list = np.array(val_loss_list)
        self.plot_train()

        pretrained_params = params
        return pretrained_params


def previous_network_pretrain(
    key: jax.Array,
    flow: hk.Transformed,
    x_init: jax.Array,
    z_target: jax.Array,
    iterations: int,
    tolerance: float = 1e-6,
) -> optax_base.Params:
    """Pretrain the network
    NOTE: a specific implementation that pretrain the normalizing flow
        such that the transformed coordinates z_transed from x_init:
        z_transed = flow(x_init)
        is equal to z_target.
    NOTE that z is the coordinates in latent space and x is the real
        configuration space.

    Args:
        key: the PRNGKey.
        flow: the flow model which is transformed into pure function
            by haiku.
        x_init: (...) the initial coordinates in configuration space
            from which the main training programm would start from
            (typically where the MCMC starts from.)
        z_target: (...) have the same shape as x_init
            the target coordinates that one would like the network to
            link with x_init. After pretrain, we would obtain a network
            that would transform x_init into z_target.
        iterations: the attempt iterations for pretrain. If after
            iterations times of pretrain and tolerance is not reached,
            raise error.
        tolerance: the tolerance of convergence.
            NOTE: the criteria of convergence is 100 times of hitting the
                tolerance.

    Returns:
        pretrained_params: the network parameters after pretrain. Using this
            parameter directly draws result that flow(x_init) = z_target.
    """
    converge_criteria = 10

    def _loss(params: optax_base.Params) -> jax.Array:
        """Loss function"""
        z_transed = flow.apply(params, None, x_init)
        return jnp.linalg.norm(z_transed - z_target)

    key, subkey = jax.random.split(key)

    init_params = flow.init(subkey, x_init)

    init_lr = 2e-2
    optimizer = optax.adam(init_lr)
    opt_state = optimizer.init(init_params)
    params = init_params

    convergence_count = 0
    for i in range(iterations):
        loss_i = _loss(params)
        print(f"Epoch: {i:06}, Loss={loss_i:.5f}")

        if loss_i <= tolerance:
            convergence_count += 1
        else:
            convergence_count = 0

        if convergence_count >= converge_criteria:
            break

        grad = jax.grad(_loss)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    if convergence_count < converge_criteria:
        raise TimeoutError(
            f"Convergence failed to reach with tolerance={tolerance}"
            f" within iterations={iterations} by hitting tolerance "
            f"{converge_criteria} times contineously."
        )

    pretrained_params = params
    return pretrained_params

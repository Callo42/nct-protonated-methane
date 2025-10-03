"""Test energy_estimator.py"""

import unittest
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np

from neuralvib.wfbasis.wf_ansatze import WFAnsatz
from neuralvib.wfbasis.basis import InvariantHermiteFunction
from neuralvib.utils.energy_estimator import EnergyEstimator
from neuralvib.molecule.utils.init_molecule import InitMolecule


class PseudoArg(Namespace):
    """A pseudo argparse.Namespace for EnergyEstimator test

    Attributes:
        select_potential: the selected potential, used in EnergyEstimator test.
    """

    def __init__(self):
        super().__init__()
        self.select_potential = "J.Phys.Chem.A2006,110,1569-1574"


def pseudo_wf_ansatz(
    params: dict, x: jax.Array | np.ndarray, excitation_number: np.ndarray | jax.Array
) -> jax.Array:
    """Pseudo wavefunction for test
    NOTE: in log domain

    Args:
        params: the pseudo network parameters, ONLY for fitting
            the argument calling in EnergyEstimator here.
        x: (num_of_particles,dim,) the configuration cartesian
            coordinates
        excitation_number: (num_of_particles*dim,) the corresponding excitation
            quantum number of each 1d-oscillator (of each 1d coordinate),
             in the same order as that in coors(flattened).
    Returns:
        log_wf: the wavefunction in log
            domain
    """
    square_x = (x**2).sum(axis=-1)
    log_wf = jnp.sum(square_x)
    return log_wf


def pseudo_pes(x: jax.Array | np.ndarray) -> jax.Array:
    """Pseudo potential function for test

    Args:
        x: (num_of_particles,dim,) the configuration cartesian
            coordinates
    Returns:
        potential: the potential energy of corresponding config
    """
    potential = jnp.sum(x**2) / 3
    return potential


def pseudo_pes_for_joblib(x: np.ndarray) -> np.ndarray:
    """Pseudo potential function for test
    For joblib parallel.

    Args:
        x: (num_of_particles,dim,) the configuration cartesian
            coordinates
    Returns:
        potential: the potential energy of corresponding config
    """
    potential = np.sum(x**2) / 3
    return potential


class TestEnergyEstimator(unittest.TestCase):
    """Test EnergyEstimator"""

    def setUp(self) -> None:
        """SetUp"""
        self.energy_estimator = EnergyEstimator(
            wf_ansatz=pseudo_wf_ansatz, potential_func=pseudo_pes, particle_mass=1.0
        )
        self.excitation_number = np.array([0] * 3)

    def tearDown(self) -> None:
        """TearDown"""
        del self.energy_estimator

    def test_kinetic(self) -> None:
        """test kinetic estimator"""
        kinetic_estimator = self.energy_estimator.local_kinetic_energy
        pseudo_params = {"w": 1.0}
        x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        expected_kinetic = -0.5 * (12 + 4 * np.sum(x**2))
        local_kinetic = kinetic_estimator(
            params=pseudo_params, x=x, excitation_number=self.excitation_number
        )
        print(f"local_kinetic={local_kinetic}")
        np.testing.assert_almost_equal(expected_kinetic, local_kinetic)

    def test_local_energy(self) -> None:
        """test local energy estimator"""
        local_energy_estimator = self.energy_estimator.local_energy
        pseudo_params = {"w": 1.0}
        x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        expected_energy = -0.5 * (12 + 4 * np.sum(x**2)) + pseudo_pes(x)
        local_energy = local_energy_estimator(
            params=pseudo_params, x=x, excitation_number=self.excitation_number
        )
        print(f"local_energy={local_energy}")
        np.testing.assert_almost_equal(expected_energy, local_energy)

    def test_batched_local_energy(self) -> None:
        """test batched local energy estimator"""
        batched_local_energy_estimator = self.energy_estimator.vmap_batched_local_energy
        pseudo_params = {"w": 1.0}
        x = np.array(
            [
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 1.0, 1.3], [2.4, 1.4, 2.6]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        expected_energy = []
        for single_sample in x:
            expected_energy.append(
                -0.5 * (12 + 4 * np.sum(single_sample**2)) + pseudo_pes(single_sample)
            )
        expected_energy = np.array(expected_energy)
        local_energy = batched_local_energy_estimator(
            params=pseudo_params,
            batched_x=x,
            excitation_number=self.excitation_number,
        )
        print(f"local_energy={local_energy}")
        np.testing.assert_array_almost_equal(expected_energy, local_energy, decimal=5)

    def test_local_energies(self) -> None:
        """test local energy estimator
        in seperated version
        """
        local_energy_estimator = self.energy_estimator.local_energies
        pseudo_params = {"w": 1.0}
        x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        expected_kinetic = -0.5 * (12 + 4 * np.sum(x**2))
        expected_potential = pseudo_pes(x)
        kin_energy, pot_energy = local_energy_estimator(
            params=pseudo_params, x=x, excitation_number=self.excitation_number
        )
        print(f"kin_energy={kin_energy}, pot_energy={pot_energy}")
        np.testing.assert_almost_equal(expected_kinetic, kin_energy)
        np.testing.assert_almost_equal(expected_potential, pot_energy)

    def test_batched_local_energies(self) -> None:
        """test batched local energy estimator
        in seperated version
        """
        batched_local_energy_estimator = (
            self.energy_estimator.vmap_batched_local_energies
        )
        pseudo_params = {"w": 1.0}
        x = np.array(
            [
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 1.0, 1.3], [2.4, 1.4, 2.6]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        expected_kinetics = []
        expected_potentials = []
        for single_sample in x:
            expected_kinetics.append(-0.5 * (12 + 4 * np.sum(single_sample**2)))
            expected_potentials.append(pseudo_pes(single_sample))
        expected_kinetics = np.array(expected_kinetics)
        expected_potentials = np.array(expected_potentials)
        batchde_kinetics, batched_potentials = batched_local_energy_estimator(
            params=pseudo_params,
            batched_x=x,
            excitation_number=self.excitation_number,
        )
        print(
            f"batched_kinetics={batchde_kinetics}\n"
            f"batched_potentials={batched_potentials}"
        )
        np.testing.assert_array_almost_equal(
            expected_kinetics, batchde_kinetics, decimal=5
        )
        np.testing.assert_array_almost_equal(
            expected_potentials, batched_potentials, decimal=5
        )


class TestEnergyEstimatorJoblib(unittest.TestCase):
    """Test EnergyEstimator with joblib paralleled."""

    def setUp(self) -> None:
        """SetUp"""
        self.energy_estimator = EnergyEstimator(
            wf_ansatz=pseudo_wf_ansatz,
            potential_func=pseudo_pes_for_joblib,
            particle_mass=1.0,
            pot_batch_method="joblib",
            seperate_energy=True,
        )
        self.excitation_number = np.array([0] * 3)

    def tearDown(self) -> None:
        """TearDown"""
        del self.energy_estimator

    def test_local_energy(self) -> None:
        """test local energy estimator"""
        local_energy_estimator = self.energy_estimator.local_energy
        pseudo_params = {"w": 1.0}
        x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        expected_energy = -0.5 * (12 + 4 * np.sum(x**2)) + pseudo_pes(x)
        local_energy = local_energy_estimator(
            params=pseudo_params, x=x, excitation_number=self.excitation_number
        )
        print(f"local_energy={local_energy}")
        np.testing.assert_almost_equal(expected_energy, local_energy)

    def test_batched_local_energy(self) -> None:
        """test batched local energy estimator"""
        batched_local_energy_estimator = (
            self.energy_estimator.joblib_batched_local_energy
        )
        pseudo_params = {"w": 1.0}
        x = np.array(
            [
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 1.0, 1.3], [2.4, 1.4, 2.6]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        expected_energy = []
        for single_sample in x:
            expected_energy.append(
                -0.5 * (12 + 4 * np.sum(single_sample**2)) + pseudo_pes(single_sample)
            )
        expected_energy = np.array(expected_energy)
        local_energy = batched_local_energy_estimator(
            params=pseudo_params,
            batched_x=x,
            excitation_number=self.excitation_number,
        )
        print(f"local_energy={local_energy}")
        np.testing.assert_array_almost_equal(expected_energy, local_energy, decimal=5)

    def test_batched_local_energies(self) -> None:
        """test batched local energy estimator
        in seperated version
        """
        batched_local_energy_estimator = (
            self.energy_estimator.joblib_batched_local_energies
        )
        pseudo_params = {"w": 1.0}
        x = np.array(
            [
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 1.0, 1.3], [2.4, 1.4, 2.6]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        expected_kinetics = []
        expected_potentials = []
        for single_sample in x:
            expected_kinetics.append(-0.5 * (12 + 4 * np.sum(single_sample**2)))
            expected_potentials.append(pseudo_pes(single_sample))
        expected_kinetics = np.array(expected_kinetics)
        expected_potentials = np.array(expected_potentials)
        batchde_kinetics, batched_potentials = batched_local_energy_estimator(
            params=pseudo_params, batched_x=x, excitation_number=self.excitation_number
        )
        print(
            f"batched_kinetics={batchde_kinetics}\n"
            f"batched_potentials={batched_potentials}"
        )
        np.testing.assert_array_almost_equal(
            expected_kinetics, batchde_kinetics, decimal=5
        )
        np.testing.assert_array_almost_equal(
            expected_potentials, batched_potentials, decimal=5
        )


class TestEnergyEstimatorHermiteFunction(unittest.TestCase):
    """Test energy estimator upon (real) Hermite functions"""

    def setUp(self) -> None:
        """SetUp"""
        psudo_args = PseudoArg()
        molecule_init_obj = InitMolecule(
            molecule="CH5+",
            x_alpha=1.0,
            input_args=psudo_args,
        )
        invariant_hermite_func_obj = InvariantHermiteFunction(
            particles=molecule_init_obj.particles,
            m=molecule_init_obj.particle_mass,
            w=molecule_init_obj.omega,
        )

        def _log_wf_ansatz(params, x, excitation_number):
            return invariant_hermite_func_obj.log_phi_base(x, excitation_number)

        self.energy_estimator = EnergyEstimator(
            wf_ansatz=_log_wf_ansatz,
            potential_func=molecule_init_obj.pes_cartesian,
            particle_mass=molecule_init_obj.particle_mass,
            pot_batch_method=molecule_init_obj.pot_parallel_method,
        )
        self.molecule_init_obj = molecule_init_obj

    def tearDown(self) -> None:
        """TearDown"""
        del self.energy_estimator

    def expect_kinetic(self, x):
        """
        Calculate the expected kinetic energy for a given configuration.

        Args:
            x (np.ndarray): The configuration of the system.

        Returns:
            float: The expected kinetic energy.
        """
        x = x.reshape(-1)
        m = self.molecule_init_obj.particle_mass
        w = self.molecule_init_obj.omega
        kinetics = -0.5 * w * (m * w * x**2 - 1)
        return np.sum(kinetics)

    def test_kinetic(self) -> None:
        """Test local kinetic"""
        kinetic_operator = self.energy_estimator.local_kinetic_energy
        params = {"pseudo": 0.0}
        x = np.random.normal(size=(5, 3))
        excitation_number = np.zeros(15, dtype=int)
        expected_local_kinetic = self.expect_kinetic(x)
        real_local_kinetic = kinetic_operator(params, x, excitation_number)
        np.testing.assert_almost_equal(real_local_kinetic, expected_local_kinetic)

    def test_batched_local_energies(self) -> None:
        """Test batched local energies for both vmap and joblib methods"""
        params = {"pseudo": 0.0}
        batch_size = 3
        batched_x = np.random.normal(size=(batch_size, 5, 3))
        excitation_number = np.zeros(15, dtype=int)

        # Test vmap batched energies
        vmap_energy_estimator = EnergyEstimator(
            wf_ansatz=self.energy_estimator.wf_ansatz,
            potential_func=self.molecule_init_obj.pes_cartesian,
            particle_mass=self.molecule_init_obj.particle_mass,
            pot_batch_method="jax.vmap",
            seperate_energy=True,
        )
        vmap_kinetics, vmap_potentials = vmap_energy_estimator.batched_local_energy(
            params, batched_x, excitation_number
        )

        # Verify each batch element against non-batched calculation
        for i in range(batch_size):
            expected_kinetic = self.expect_kinetic(batched_x[i])
            expected_potential = self.molecule_init_obj.pes_cartesian(
                np.concatenate((np.array([[0.0, 0.0, 0.0]]), batched_x[i]))
            )

            np.testing.assert_almost_equal(vmap_kinetics[i], expected_kinetic)
            np.testing.assert_almost_equal(vmap_potentials[i], expected_potential)


if __name__ == "__main__":
    unittest.main()

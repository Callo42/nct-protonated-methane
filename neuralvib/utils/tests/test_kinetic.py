"""Test kinetic energy estimator"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from neuralvib.utils.energy_estimator import EnergyEstimator


def simple_wf_ansatz(params, x, excitation_number):
    """Simple test wavefunction: Gaussian centered at origin"""
    return -0.5 * jnp.sum(x**2)  # log domain


def constant_wf_ansatz(params, x, excitation_number):
    """Constant wavefunction (edge case)"""
    return jnp.array(0.0)  # log domain = ln(1)


def linear_wf_ansatz(params, x, excitation_number):
    """Linear wavefunction (another edge case)"""
    return jnp.sum(x)  # log domain


def dummy_potential(x):
    """Dummy potential function"""
    return jnp.sum(x**2) / 2


class TestKineticEstimator(unittest.TestCase):
    """Test LOCAL kinetic energy estimator"""

    def setUp(self):
        """Setup test cases"""
        self.particles = ("H", "H")  # Two hydrogen atoms
        self.particle_mass = {"H": 1.0}  # Simplified mass for testing

        # Initialize estimator with gaussian wavefunction
        self.estimator = EnergyEstimator(
            wf_ansatz=simple_wf_ansatz,
            potential_func=dummy_potential,
            particles=self.particles,
            particle_mass=self.particle_mass,
        )

        # Initialize estimators with edge case wavefunctions
        self.constant_estimator = EnergyEstimator(
            wf_ansatz=constant_wf_ansatz,
            potential_func=dummy_potential,
            particles=self.particles,
            particle_mass=self.particle_mass,
        )

        self.linear_estimator = EnergyEstimator(
            wf_ansatz=linear_wf_ansatz,
            potential_func=dummy_potential,
            particles=self.particles,
            particle_mass=self.particle_mass,
        )

    def test_gaussian_ground_state(self):
        """Test kinetic energy for Gaussian ground state
        For harmonic oscillator ground state, <T> = ħω/4 = 1/2 (in atomic units)
        """
        params = {}
        x = jnp.zeros((2, 3))  # Two particles in 3D at origin
        excitation_number = jnp.zeros(6)

        kinetic = self.estimator.local_kinetic_energy(params, x, excitation_number)
        expected = 3.0  # 3D harmonic oscillator

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_gaussian_displaced(self):
        """Test kinetic energy for displaced Gaussian"""
        params = {}
        x = jnp.ones((2, 3))  # Two particles in 3D, displaced from origin
        excitation_number = jnp.zeros(6)

        kinetic = self.estimator.local_kinetic_energy(params, x, excitation_number)
        expected = 0.0

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_constant_wavefunction(self):
        """Test kinetic energy for constant wavefunction (edge case)
        Kinetic energy should be zero for constant wavefunction
        """
        params = {}
        x = jnp.zeros((2, 3))
        excitation_number = jnp.zeros(6)

        kinetic = self.constant_estimator.local_kinetic_energy(
            params, x, excitation_number
        )
        expected = 0.0

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_linear_wavefunction(self):
        """Test kinetic energy for linear wavefunction (edge case)
        For ψ = exp(x), kinetic energy should be constant
        """
        params = {}
        x = jnp.ones((2, 3))
        excitation_number = jnp.zeros(6)

        kinetic = self.linear_estimator.local_kinetic_energy(
            params, x, excitation_number
        )
        expected = -3.0  # -0.5 * (0 + 1^2) * 6 dimensions

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_different_masses(self):
        """Test kinetic energy with different particle masses"""
        estimator = EnergyEstimator(
            wf_ansatz=simple_wf_ansatz,
            potential_func=dummy_potential,
            particles=("H", "D"),  # Hydrogen and Deuterium
            particle_mass={"H": 1.0, "D": 2.0},
        )

        params = {}
        x = jnp.zeros((2, 3))
        excitation_number = jnp.zeros(6)

        kinetic = estimator.local_kinetic_energy(params, x, excitation_number)
        expected = 2.25  # 3/2 + 3/4 for different masses

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_no_kinetic(self):
        """Test with no_kinetic flag set to True"""
        estimator = EnergyEstimator(
            wf_ansatz=simple_wf_ansatz,
            potential_func=dummy_potential,
            particles=self.particles,
            particle_mass=self.particle_mass,
            no_kinetic=True,
        )

        params = {}
        x = jnp.ones((2, 3))
        excitation_number = jnp.zeros(6)

        kinetic = estimator.local_kinetic_energy(params, x, excitation_number)
        expected = 0.0

        np.testing.assert_almost_equal(kinetic, expected, decimal=5)

    def test_numerical_stability(self):
        """Test numerical stability with large displacements"""
        params = {}
        x = 1e5 * jnp.ones((2, 3))  # Very large displacements
        excitation_number = jnp.zeros(6)

        kinetic = self.estimator.local_kinetic_energy(params, x, excitation_number)
        self.assertFalse(jnp.isnan(kinetic))
        self.assertFalse(jnp.isinf(kinetic))


if __name__ == "__main__":
    unittest.main()

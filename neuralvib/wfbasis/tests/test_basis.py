"""Test basis.py"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.wfbasis.basis import InvariantGaussian, gaussian
from neuralvib.wfbasis.basis import log_gaussian


class TestBasis(unittest.TestCase):
    """Test wavefunction basis"""

    def setUp(self) -> None:
        """Setup"""

    def tearDown(self) -> None:
        """TearDown"""

    def test_gaussian(self) -> None:
        """Test gaussian wavefunction basis"""

        # TestCase1
        x0 = jnp.array([0.0, 1.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        expect = 1 / (2 * jnp.pi) ** (3 / 2)
        real = gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

        # TestCase2
        x0 = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        expect = jnp.exp(-1 / 2) / (2 * jnp.pi) ** (3 / 2)
        real = gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

        # TestCase3
        x0 = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([2.0, 1.0, 0.0])
        r2 = jnp.sum((x - x0) ** 2)
        expect = jnp.exp(-r2 / 2) / (2 * jnp.pi) ** (3 / 2)
        real = gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

    def test_log_gaussian(self) -> None:
        """Test log gaussian wavefunction basis"""

        # TestCase1
        x0 = jnp.array([0.0, 1.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        expect = -(3 / 2) * jnp.log(2 * jnp.pi)
        real = log_gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

        # TestCase2
        x0 = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        expect = -1 / 2 - (3 / 2) * jnp.log(2 * jnp.pi)
        real = log_gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

        # TestCase3
        x0 = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        r2 = jnp.sum((x - x0) ** 2)
        expect = -r2 / 2 - (3 / 2) * jnp.log(2 * jnp.pi)
        real = log_gaussian(x, x0)
        self.assertAlmostEqual(expect, real)

        # TestCase3
        x0 = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 0.0])
        sigma = 2.0
        r2 = jnp.sum((x - x0) ** 2)
        expect = -r2 / (2 * sigma**2) - (1 / 2) * jnp.log((2 * jnp.pi) ** 3 * sigma**6)
        real = log_gaussian(x, x0, sigma)
        self.assertAlmostEqual(expect, real)

    def test_invariant_gaussian(self) -> None:
        """Test log invariant gaussian"""
        particles = ("C", "H", "H", "H", "H", "H")
        partition = np.array([1])
        sigma_choice = 2.0
        print(f"============sigma={sigma_choice}===================")
        sigma_carbon = sigma_choice
        sigma_hydrogen = sigma_choice
        sigmas = np.array([sigma_carbon] + [sigma_hydrogen] * 5)
        invaraint_gaussian_obj = InvariantGaussian(
            particles=particles,
            partition=partition,
            sigmas=sigmas,
        )
        log_invariant_gaussian = invaraint_gaussian_obj.log_invariant_gaussian

        hydrogen_permutation = np.random.permutation(5)
        config_permutation = np.concatenate(
            (
                np.array([0]),
                hydrogen_permutation + 1,
            )
        )
        coordinates = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        # coordinates = np.array(
        #     [[ 0.,          0.,          0.,        ],
        #         [ 0.,          1.1,          0.,        ],
        #         [ 0.9,  0.3,  0.        ],
        #         [ 0.5, -0.8,  0.        ],
        #         [-0.5, -0.8,  0.        ],
        #         [-0.9,  0.3,  0.        ]]
        # )
        coordinates_after_permut = coordinates[config_permutation]

        log_phi_base_origin = log_invariant_gaussian(coordinates)
        log_phi_base_permutated = log_invariant_gaussian(coordinates_after_permut)
        print(f"============sigma={sigma_choice}===================")
        self.assertAlmostEqual(log_phi_base_origin, log_phi_base_permutated)

    def test_invariant_gaussian_mixture(self) -> None:
        """Test log invariant gaussian mixture"""
        particles = ("H", "H", "H", "H", "H")
        partition = np.array([0])
        sigma_choice = 2.0
        print(f"============sigma={sigma_choice}===================")
        sigma_hydrogen = sigma_choice
        sigmas = np.array([sigma_hydrogen] * 5)
        invaraint_gaussian_obj = InvariantGaussian(
            particles=particles,
            partition=partition,
            sigmas=sigmas,
            symmetry_implement="use_partition",
        )
        log_invariant_gaussian = invaraint_gaussian_obj.log_invariant_gaussian_mixture

        hydrogen_permutation = np.random.permutation(5)
        config_permutation = hydrogen_permutation
        coordinates = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)[1::, :]
        # coordinates = np.array(
        #     [[ 0.,          0.,          0.,        ],
        #         [ 0.,          1.1,          0.,        ],
        #         [ 0.9,  0.3,  0.        ],
        #         [ 0.5, -0.8,  0.        ],
        #         [-0.5, -0.8,  0.        ],
        #         [-0.9,  0.3,  0.        ]]
        # )
        coordinates_after_permut = coordinates[config_permutation]

        log_phi_base_origin = log_invariant_gaussian(coordinates)
        log_phi_base_permutated = log_invariant_gaussian(coordinates_after_permut)
        print(f"============sigma={sigma_choice}===================")
        self.assertAlmostEqual(log_phi_base_origin, log_phi_base_permutated)


if __name__ == "__main__":
    unittest.main()

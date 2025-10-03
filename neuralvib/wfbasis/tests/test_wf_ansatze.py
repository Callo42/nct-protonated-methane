"""Test wf_ansatze.py"""

import unittest

import numpy as np
import haiku as hk
import jax

from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.networks.flow_MoleNet import MoleNetFlow
from neuralvib.wfbasis.basis import InvariantGaussian
from neuralvib.wfbasis.basis import InvariantHermiteFunction
from neuralvib.wfbasis.wf_ansatze import WFAnsatz
from neuralvib.utils.convert import convert_inverse_cm_to_hartree


class TestWFAnsatz(unittest.TestCase):
    """Test wavefunction basis"""

    def setUp(self) -> None:
        """Setup"""

    def tearDown(self) -> None:
        """TearDown"""

    def test_invariant_hermite_functions(self) -> None:
        """Test log invariant hermite functions"""
        key = jax.random.PRNGKey(42)
        particles = ("H", "H", "H", "H", "H")
        mass = 1836.152673
        zpe = convert_inverse_cm_to_hartree(10917.0)
        omega = 2 * zpe / 15
        excitation_number = np.zeros(15, dtype=int)
        invaraint_gaussian_obj = InvariantHermiteFunction(
            particles=particles,
            m=mass,
            w=omega,
        )
        log_invariant_gaussian = invaraint_gaussian_obj.log_phi_base

        hydrogen_permutation = np.random.permutation(5)
        config_permutation = hydrogen_permutation
        coordinates = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)[1::]
        coordinates_after_permut = coordinates[config_permutation]

        def flow_fn(x):
            model = MoleNetFlow(
                depth=2,
                h1_size=4,
                h2_size=4,
                partitions=[1],
            )
            return model(x)

        flow = hk.transform(flow_fn)
        params = flow.init(key, coordinates)

        wf_ansatze_obj = WFAnsatz(flow=flow, log_phi_base=log_invariant_gaussian)

        log_wf_ansatze = wf_ansatze_obj.log_wf_ansatz

        log_psi_origin = log_wf_ansatze(params, coordinates, excitation_number)
        log_psi_permutated = log_wf_ansatze(
            params, coordinates_after_permut, excitation_number
        )
        print(
            f"log_psi_origin={log_psi_origin}\nlog_psi_permutated={log_psi_permutated}"
        )
        np.testing.assert_array_almost_equal(
            log_psi_origin, log_psi_permutated, decimal=5
        )


if __name__ == "__main__":
    unittest.main()

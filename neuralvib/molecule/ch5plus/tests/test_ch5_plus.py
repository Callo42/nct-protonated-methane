"""Test CH5+ Molecule"""

import unittest

import numpy as np

from neuralvib.molecule.ch5plus.ch5_plus import CH5Plus
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.convert import convert_hartree_to_inverse_cm


class TestCH5MoleculeCartesian(unittest.TestCase):
    """Test CH5 plus Molecule in configuration cartesian coordinates."""

    def setUp(self):
        ch5 = CH5Plus(select_potential="J.Phys.Chem.A2006,110,1569-1574")
        ch5.xalpha = 1.0
        self.potential_func = ch5.pes_config_cartesian
        self.hartree2cm_coeff = convert_hartree_to_inverse_cm(1.0)

        # Global Minimun
        self.cs1_cartesian = equilibrium_bowman_jpca_2006_110_1569_1574()
        self.expect_minimum = 0.0

        #  C2v minimum for bond vectors
        self.cs2_cartesian = saddle_c2v_bowman_jpca_2006_110_1569_1574()
        self.expect_c2v = 340.759

        # test for rejection
        self.vceil = 0.09112669999999999  # 20000 cm-1
        self.rejection_config = np.array(
            [
                [0, 0, 0],
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, -0.5, 0],
                [-0.5, 0, 0],
                [0, 0, 0.5],
            ]
        )

    def tearDown(self):
        pass

    def test_potential(self):
        """Test cartesian potential."""
        pot1 = self.potential_func(self.cs1_cartesian)
        pot1 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(self.expect_minimum, pot1, decimal=4)

        pot2 = self.potential_func(self.cs2_cartesian)
        pot2 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot2, self.expect_c2v, decimal=3)

    def test_potential_reject(self):
        """Test pes function rejection"""
        pot1 = self.potential_func(self.rejection_config)
        np.testing.assert_almost_equal(
            pot1,
            self.vceil,
        )


if __name__ == "__main__":
    unittest.main()

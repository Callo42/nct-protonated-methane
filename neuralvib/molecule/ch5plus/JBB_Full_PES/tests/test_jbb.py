"""Test JBB Potential PES"""

import time
import unittest
import os

import numpy as np
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.JBB_Full_PES import JBBCH5ppotential
from neuralvib.molecule.ch5plus.JBB_Full_PES.jbbjax import (
    jbbf2py_pes_cartesian,
    jbbf2py_pes_ch5ppot_cart,
    jbbjax_ch5ppot_cart,
    jbbjax_polar,
)
from neuralvib.molecule.ch5plus.JBB_Full_PES.jbbjax import jbbjax_cartesian
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.convert import convert_hartree_to_inverse_cm


class TestJBBPESPolar(unittest.TestCase):
    """Test JBB PES for ch5 plus in polar coordinates."""

    def setUp(self):
        self.potential_func = JBBCH5ppotential.ch5ppot_func

        # Optimized geometry for bond vectors
        r_list = (
            "    2.09459     2.05676     2.26113     2.26260     2.05676"
            " 109.248     119.433      97.451     116.374"
            " 126.722     229.665     274.835"
        )
        r = np.array(r_list.split(), dtype=np.float64)
        r[5::] *= np.radians(1.0)
        self.r_minimum = r
        self.expect_minimum = 0.0

        #  C2v minimum for bond vectors
        r2_list = (
            "2.15785     2.05472     2.05472     2.15785     2.19719"
            " 61.566     118.973     118.973      61.566"
            " 90.000     270.000     180.000"
        )
        r2 = np.array(r2_list.split(), dtype=np.float64)
        r2[5::] *= np.radians(1.0)
        self.r_c2v = r2
        self.expect_c2v = 340.759

    def tearDown(self):
        pass

    def test_potential(self):
        """Test the f2py function
        performs the same as original
        fortran ones.
        """
        pot1 = self.potential_func(self.r_minimum)
        np.testing.assert_almost_equal(self.expect_minimum, pot1, decimal=3)

        pot2 = self.potential_func(self.r_c2v)
        np.testing.assert_almost_equal(self.expect_c2v, pot2, decimal=3)

    def test_jaxjbb(self):
        """Test jax called back function"""
        pot1 = jbbjax_polar(self.r_minimum)
        # print(f"pot1={pot1}")
        np.testing.assert_almost_equal(self.expect_minimum, pot1, decimal=3)
        pot2 = jbbjax_polar(self.r_c2v)
        np.testing.assert_almost_equal(self.expect_c2v, pot2, decimal=3)

    def test_jaxjbb_vmap(self):
        """Test jax called back function"""
        # rs = np.array(
        #     [
        #         self.r_minimum,
        #         self.r_c2v,
        #     ]
        # )
        # desired_pots = np.array(
        #     [
        #         self.expect_minimum,
        #         self.expect_c2v,
        #     ]
        # )
        # jbb_vmapped = jax.vmap(jbbjax_vec)
        # actual_pots = jbb_vmapped(rs)
        # print(desired_pots)
        # print(actual_pots)
        # np.testing.assert_allclose(actual_pots, desired_pots)


class TestJBBPESCartesian(unittest.TestCase):
    """Test JBB PES for ch5 plus in cartesian coordinates."""

    def setUp(self):
        self.potential_func = jbbf2py_pes_cartesian
        self.hartree2cm_coeff = convert_hartree_to_inverse_cm(1.0)

        # Global Minimun
        self.cs1_cartesian = jnp.array(
            equilibrium_bowman_jpca_2006_110_1569_1574()
        ).reshape(6, 3)
        self.expect_minimum = 0.0

        #  C2v minimum for bond vectors
        self.cs2_cartesian = jnp.array(
            saddle_c2v_bowman_jpca_2006_110_1569_1574()
        ).reshape(6, 3)
        self.expect_c2v = 340.759

        # test for rejection
        # self.vceil = 0.05467602 # 12000 cm-1
        self.vceil = 0.09112669999999999  # 20000 cm-1
        self.rejection_config = jnp.array(
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
        """Test the f2py function
        performs the same as original
        fortran ones.
        """
        pot1 = self.potential_func(self.cs1_cartesian.T)
        pot1 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot1, self.expect_minimum, decimal=4)

        pot2 = self.potential_func(self.cs2_cartesian.T)
        pot2 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot2, self.expect_c2v, decimal=3)

    def test_jaxjbb(self):
        """Test jax called back function"""

        print("Testing PES Call time cost...")
        t0 = time.time()
        pot1 = jbbjax_cartesian(self.cs1_cartesian.T)
        t1 = time.time()
        print(f"Single PES call: {t1-t0}s")

        pot1 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot1, self.expect_minimum, decimal=4)

        t0 = time.time()
        pot2 = jbbjax_cartesian(self.cs2_cartesian.T)
        t1 = time.time()
        print(f"Single PES call: {t1-t0}s")

        pot2 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot2, self.expect_c2v, decimal=3)

        for i in range(5):
            t0 = time.time()
            pot2 = jbbjax_cartesian(self.cs2_cartesian.T)
            t1 = time.time()
            print(f"Single PES call: {t1-t0}s")

    def test_jaxjbb_reject(self):
        """Test jax called back function rejection"""
        pot1 = jbbjax_cartesian(self.rejection_config.T)
        np.testing.assert_almost_equal(
            pot1,
            self.vceil,
        )


class TestJBBPESCh5ppotCart(unittest.TestCase):
    """Test JBB PES for ch5 plus: ch5ppot_func_cart in cartesian coordinates
    of 5 hydrogens.
    """

    def setUp(self):
        self.potential_func = jbbf2py_pes_ch5ppot_cart
        self.jaxpes = jbbjax_ch5ppot_cart
        self.hartree2cm_coeff = convert_hartree_to_inverse_cm(1.0)

        # Global Minimun
        self.cs1_cartesian = jnp.array(
            equilibrium_bowman_jpca_2006_110_1569_1574()
        ).reshape(6, 3)
        self.expect_minimum = 0.0
        self.cartr1 = self.cs1_cartesian[1::].T

        #  C2v minimum for bond vectors
        self.cs2_cartesian = jnp.array(
            saddle_c2v_bowman_jpca_2006_110_1569_1574()
        ).reshape(6, 3)
        self.expect_c2v = 340.759
        self.cartr2 = self.cs2_cartesian[1::].T

    def tearDown(self):
        pass

    def test_potential(self):
        """Test the f2py function
        performs the same as original
        fortran ones.
        """
        pot1 = self.potential_func(self.cartr1)
        pot1 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(self.expect_minimum, pot1, decimal=4)

        pot2 = self.potential_func(self.cartr2)
        pot2 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot2, self.expect_c2v, decimal=3)

    def test_jaxjbb(self):
        """Test jax called back function"""
        pot1 = self.jaxpes(self.cartr1)
        pot1 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot1, self.expect_minimum, decimal=4)
        pot2 = self.jaxpes(self.cartr2)
        pot2 *= self.hartree2cm_coeff
        np.testing.assert_almost_equal(pot2, self.expect_c2v, decimal=3)


if __name__ == "__main__":
    unittest.main()

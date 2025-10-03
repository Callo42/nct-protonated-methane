"""Test coordinate conversion"""

# %%
from itertools import combinations
import unittest

import numpy as np
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.JBB_Full_PES.convert_coors import (
    config_cartesian2jbb_cartesian_input_xn,
    config_cartesian2jbb_ch5ppot_cart_input,
    config_cartesian2jbb_polar_input,
    _calculate_angle_0_pi,
)
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)


class TestCoorConversionCartr(unittest.TestCase):
    """Test the coordinate conversion to JBB
    PES ch5ppot_func_cartr input for ch5 plus.
    NOTE: this is the test for (3,5) shaped
    input cartr to pes ch5ppot_func_cartr.
    """

    def setUp(self):
        # C2v saddle point, in (3,5)
        self.desired = np.array(
            [
                [
                    1.89754030e00,
                    1.10069284e-16,
                    -3.30207853e-16,
                    -1.89754030e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    1.79756783e00,
                    -1.79756783e00,
                    2.32381666e-16,
                    0.00000000e00,
                ],
                [
                    1.02745190e00,
                    -9.95301050e-01,
                    -9.95301050e-01,
                    1.02745190e00,
                    2.19719000e00,
                ],
            ]
        )
        self.saddle_c2v = jnp.array(saddle_c2v_bowman_jpca_2006_110_1569_1574())

    def tearDown(self):
        pass

    def test_equilibrium(self):
        """Test the f2py for equilibrium configuration"""
        # test generate equiblibrium
        np.testing.assert_array_almost_equal(
            config_cartesian2jbb_ch5ppot_cart_input(self.saddle_c2v),
            self.desired,
        )


class TestCoorConversionXn(unittest.TestCase):
    """Test the coordinate conversion to JBB
    PES input for ch5 plus.
    NOTE: this is the test for (3,6) shaped
    input xn to pes getpot_with_return.
    """

    def setUp(self):
        # C2v saddle point, in (3,6)
        self.desired = np.array(
            [
                [
                    0.00000000e00,
                    1.89754030e00,
                    1.10069284e-16,
                    -3.30207853e-16,
                    -1.89754030e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    1.79756783e00,
                    -1.79756783e00,
                    2.32381666e-16,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    1.02745190e00,
                    -9.95301050e-01,
                    -9.95301050e-01,
                    1.02745190e00,
                    2.19719000e00,
                ],
            ]
        )
        self.saddle_c2v = saddle_c2v_bowman_jpca_2006_110_1569_1574()

    def tearDown(self):
        pass

    def test_equilibrium(self):
        """Test the f2py for equilibrium configuration"""
        # test generate equiblibrium
        np.testing.assert_array_almost_equal(
            config_cartesian2jbb_cartesian_input_xn(self.saddle_c2v),
            self.desired,
        )


class TestCoorConversionPolar(unittest.TestCase):
    """Test the coordinate conversion to JBB
    PES input for ch5 plus.
    NOTE: this is the test for polar coordinate
    input function, ch5ppot_func.
    """

    def setUp(self):
        self.result_1 = jnp.array(
            [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                jnp.pi / 2,
                jnp.pi / 2,
                jnp.pi / 2,
                jnp.pi / 2,
                jnp.pi / 2,
                jnp.pi,
                3 * jnp.pi / 2,
            ]
        )
        self.coor_1 = jnp.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [-2, 0, 0],
                [0, -2, 0],
                [0, 0, 2],
            ]
        )

        self.equilibrium = jnp.array(
            [
                2.09459,
                2.05676,
                2.26113,
                2.2626,
                2.05676,
                1.9067373,
                2.08449909,
                1.70084081,
                2.03110946,
                2.21171613,
                4.00841043,
                4.79677565,
            ]
        )
        self.eq_conf = jnp.array(equilibrium_bowman_jpca_2006_110_1569_1574())

    def tearDown(self):
        pass

    def test_equilibrium(self):
        """Test the f2py for equilibrium configuration"""
        # test generate equiblibrium
        np.testing.assert_array_almost_equal(
            config_cartesian2jbb_polar_input(self.eq_conf),
            self.equilibrium,
        )

    def test_theta(self):
        """Test coordinate conversion
        to jbb input, the thetas.
        """
        vectors = 10 * np.random.randn(100, 3)
        for vec1, vec2 in combinations(vectors, r=2):
            theta = _calculate_angle_0_pi(vec_1=vec1, vec_2=vec2)
            print(f"vec1={vec1} and vec2={vec2}")
            self.assertTrue((0 <= theta <= np.pi))

    def test_azimuthal(self):
        """Test coordinate conversion
        to jbb input, the azimuthal, phis.
        """
        polar_coor_1 = config_cartesian2jbb_polar_input(self.coor_1)
        print(f"polar_ccor_1 = {polar_coor_1},\ntype={type(polar_coor_1)}")
        np.testing.assert_allclose(polar_coor_1, self.result_1)


if __name__ == "__main__":
    unittest.main()


# %%

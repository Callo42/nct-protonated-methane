"""Test NN PES Flax implementation"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp


from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jcp_121_4105_4116_2004,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    McCoyNNPES,
    load_flax_params,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import trans_tf_params_to_flax
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    convert_cartesian_to_sorted_cm,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import get_nn_pes_input
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_cs_ii_bowman_jcp_121_4105_4116_2004,
)

jax.config.update("jax_enable_x64", True)


class TestNNPESFlax(unittest.TestCase):
    """Test NNPES Flax implementation."""

    def setUp(self):
        self.equil_coors = equilibrium_bowman_jcp_121_4105_4116_2004()
        self.saddle = saddle_cs_ii_bowman_jcp_121_4105_4116_2004()
        #  cpl 5, 361(1970) equilibrium coors
        self.equil_coors_2 = np.array(
            [
                0,
                0,
                0,
                -0.9693,
                -1.6781,
                -0.6340,
                -0.9693,
                1.6781,
                -0.6340,
                -0.8234,
                0.0,
                2.1562,
                1.079,
                0.0,
                2.0797,
                1.9386,
                0.0,
                -0.6340,
            ]
        )
        self.model_flax = McCoyNNPES(out_dims=1)
        params_flax_file = "./neuralvib/molecule/ch5plus/McCoy_NN_PES/params_flax"
        self.params_flax = load_flax_params(filename=params_flax_file)

    def tearDown(self):
        pass

    def test_flax_nn_pes_call(self):
        """Test NN-PES Flax implementation has the same
        function call results as original tf pes from
        McCoy's
        """
        input_to_nn_pes_1 = get_nn_pes_input(
            convert_cartesian_to_sorted_cm(self.equil_coors)
        )
        input_to_nn_pes_2 = get_nn_pes_input(
            convert_cartesian_to_sorted_cm(self.equil_coors_2)
        )
        input_to_nn_pes_saddle = get_nn_pes_input(
            convert_cartesian_to_sorted_cm(self.saddle)
        )
        expect_1 = jnp.array([31.54055537], dtype=jnp.float64)
        # expect_2 = jnp.array([3273.84111299], dtype=jnp.float64)
        expect_2 = 780.72767054
        saddle_expect = 56.20240025
        result_1 = self.model_flax.apply(self.params_flax, input_to_nn_pes_1)
        result_2 = self.model_flax.apply(self.params_flax, input_to_nn_pes_2)
        result_saddle = self.model_flax.apply(self.params_flax, input_to_nn_pes_saddle)
        np.testing.assert_allclose(expect_1, result_1)
        np.testing.assert_allclose(expect_2, result_2)
        np.testing.assert_almost_equal(saddle_expect, result_saddle, decimal=5)


if __name__ == "__main__":
    unittest.main()

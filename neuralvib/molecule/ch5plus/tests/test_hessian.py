"""Test Hessian of pes_mwcd"""
# %%
import unittest
import os

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.ch5_plus import CH5Plus

jax.config.update("jax_enable_x64", True)


class TestHessian(unittest.TestCase):
    """Test NNPES Flax implementation."""

    def setUp(self):
        select_potential = "J.Phys.Chem.A2021,125,5849-5859"
        self.ch5 = CH5Plus(select_potential)
        self.w_indices_rearrange_index = self.ch5._w_indices_rearrange_index
        self.x_e = jnp.zeros(18)
        self.pes_mwcd = self.ch5.pes_mwcd

    def tearDown(self):
        pass

    def test_hessian(self):
        """Test NN-PES Flax implementation has the same
        function call results as original tf pes from
        McCoy's
        """
        hessian = jax.hessian(self.pes_mwcd)(self.x_e)
        self.assertFalse(np.isnan(hessian).any())


if __name__ == "__main__":
    unittest.main()
    from neuralvib.utils.convert import _convert_Hartree_to_inverse_cm

    hartree2cm = _convert_Hartree_to_inverse_cm(1.0)
    select_potential = "J.Phys.Chem.A2021,125,5849-5859"
    ch5 = CH5Plus(select_potential)
    w_indices = ch5.w_indices
    w_indices_cm = w_indices * hartree2cm


# %%

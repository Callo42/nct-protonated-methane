"""Test initialize.py"""

import unittest

import jax
import numpy as np

from neuralvib.utils.initialize import init_batched_x


class TestInitialize(unittest.TestCase):
    """Test initialize"""

    def setUp(self) -> None:
        """SetUp"""

    def tearDown(self) -> None:
        """TearDown"""

    def test_init_batched_x(self) -> None:
        """test kinetic estimator"""
        key = jax.random.PRNGKey(42)
        shape = (120, 5, 3)
        batch_size, num_of_particles, dim = shape
        init_x = init_batched_x(key, batch_size, num_of_particles, dim)
        np.testing.assert_allclose(init_x.shape, shape)

    def test_init_ch5_batched_x(self) -> None:
        """test kinetic estimator"""
        key = jax.random.PRNGKey(42)
        shape = (120, 6, 3)
        desired_shape = (120, 5, 3)
        batch_size, num_of_particles, dim = shape
        init_x = init_batched_x(key, batch_size, num_of_particles, dim, molecule="CH5+")
        np.testing.assert_allclose(init_x.shape, desired_shape)


if __name__ == "__main__":
    unittest.main()

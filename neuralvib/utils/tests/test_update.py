"""Test Utils"""

import unittest
import jax
import jax.numpy as jnp
from neuralvib.utils.update import clip_grad_norm
import numpy as np


class TestClipGradNorm(unittest.TestCase):
    def test_clipping_above_threshold(self):
        grads = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}  # L2 norm = 5.0
        max_norm = 3.0
        clipped = clip_grad_norm(grads, max_norm)

        expected = {"w": jnp.array([1.8, 2.4]), "b": jnp.array([0.0])}
        np.testing.assert_allclose(clipped["w"], expected["w"], rtol=1e-6)
        np.testing.assert_allclose(clipped["b"], expected["b"], rtol=1e-6)

    def test_no_clipping_below_threshold(self):
        grads = {
            "w": jnp.array([1.0, 1.0]),
            "b": jnp.array([1.0]),
        }  # L2 norm = √3 ≈ 1.732
        max_norm = 2.0
        clipped = clip_grad_norm(grads, max_norm)

        np.testing.assert_allclose(clipped["w"], grads["w"], rtol=1e-6)
        np.testing.assert_allclose(clipped["b"], grads["b"], rtol=1e-6)

    def test_exact_norm(self):
        grads = {"w": jnp.array([3.0, 4.0])}  # L2 norm = 5.0
        max_norm = 5.0
        clipped = clip_grad_norm(grads, max_norm)

        np.testing.assert_allclose(clipped["w"], grads["w"], rtol=1e-6)

    def test_zero_gradients(self):
        grads = {"w": jnp.array([0.0, 0.0]), "b": jnp.array([0.0])}
        max_norm = 1.0
        clipped = clip_grad_norm(grads, max_norm)

        np.testing.assert_allclose(clipped["w"], grads["w"], rtol=1e-6)
        np.testing.assert_allclose(clipped["b"], grads["b"], rtol=1e-6)

    def test_jit_compatibility(self):
        grad_fn = jax.jit(clip_grad_norm)
        grads = {"w": jnp.array([3.0, 4.0])}
        clipped = grad_fn(grads, 3.0)

        expected = {"w": jnp.array([1.8, 2.4])}
        np.testing.assert_allclose(clipped["w"], expected["w"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

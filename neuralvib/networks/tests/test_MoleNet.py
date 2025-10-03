import unittest

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from neuralvib.networks.flow_MoleNet import MoleNetFlow
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)

jax.config.update("jax_enable_x64", True)


class TestMoleNet(unittest.TestCase):
    """Test MoleNet for CH5+"""

    def setUp(self):
        depth = 2
        spsize, tpsize = 16, 16
        self.cartesian_coor_dim = 3
        self.key = jax.random.PRNGKey(42)
        partitions = [1]

        init_config = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        self.init_config = init_config

        def flow_fn(x):
            model = MoleNetFlow(
                depth=depth,
                h1_size=spsize,
                h2_size=tpsize,
                partitions=partitions,
            )
            return model(x)

        self.flow = hk.transform(flow_fn)
        self.params = self.flow.init(self.key, init_config)

    def tearDown(self):
        del self.flow
        del self.params

    def test_permutation_equivariance(self):
        """Test permutation equivariance of flow"""
        print("---- Test permutation equivariance ----")

        hydrogen_permutation = np.random.permutation(5)
        config_permutation = np.concatenate(
            (
                np.array([0]),
                hydrogen_permutation + 1,
            )
        )

        coordinates = self.init_config
        # F(P(z))
        before_flow_config_permutated = coordinates[config_permutation]
        flow_permute_config = self.flow.apply(
            self.params,
            None,
            before_flow_config_permutated,
        )

        # P(F(z))
        flow_config_unpermuted = self.flow.apply(self.params, None, coordinates)
        after_flow_config_permutated = flow_config_unpermuted[config_permutation]

        print(
            f"F(P(z))={flow_permute_config}\n" f"P(F(z))={after_flow_config_permutated}"
        )

        np.testing.assert_array_almost_equal(
            flow_permute_config, after_flow_config_permutated, decimal=2
        )

    def test_init_jacobian(self):
        """Test initial jacobian"""
        print("---- Test Initial Jacobian Near Identity ----")

        z = self.init_config
        num_of_particles, dim = z.shape
        fz = self.flow.apply(self.params, None, z)
        z_flatten = z.reshape(-1)

        def _flow_flatten(z_flatten):
            return self.flow.apply(
                self.params, None, z_flatten.reshape(num_of_particles, dim)
            ).reshape(-1)

        jac = jax.jacfwd(_flow_flatten)(z_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        print(f"\nx={z}")
        print(f"\nfx={fz}")
        print(f"\njac={jac}\n jac.shape={jac.shape}")
        print(f"\nlogjacdet={logjacdet}")
        np.testing.assert_array_almost_equal(
            np.eye(num_of_particles * dim), jac, decimal=3
        )

    def test_shift_equivariance(self):
        """Test shift equirariance under cartesian coordinates."""
        print("---- Test translation equivariance ----")

        dim = self.cartesian_coor_dim
        shift = np.random.randn(dim)
        z = self.init_config
        # shiftz = self.flow.apply(self.params, None, x + shift)
        # assert jnp.allclose(shiftz, z + shift)

        # F(T(z))
        before_flow_config_shifted = z + shift
        flow_shifted_z = self.flow.apply(
            self.params,
            None,
            before_flow_config_shifted,
        )

        # T(F(z))
        flow_z_unshifted = self.flow.apply(self.params, None, z)
        after_flow_z_shifted = flow_z_unshifted + shift

        print(f"F(T(z))={flow_shifted_z}\n" f"T(F(z))={after_flow_z_shifted}")

        np.testing.assert_array_almost_equal(
            flow_shifted_z, after_flow_z_shifted, decimal=3
        )


if __name__ == "__main__":
    unittest.main()

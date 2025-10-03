"""Test network_pretrain.py"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from neuralvib.networks.flow_MoleNet import MoleNetFlow
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.network_pretrain import (
    previous_network_pretrain as network_pretrain,
)


class TestPreviousNetworkPretrain(unittest.TestCase):
    """Test Previous Network Pretrain"""

    def setUp(self) -> None:
        """SetUp"""
        equil_config = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        self.x_init = equil_config[1::, :]
        self.z_target = jnp.zeros_like(self.x_init)

        depth = 4
        spsize = 32
        tpsize = 16
        partitions = [0]

        def flow_fn(x):
            model = MoleNetFlow(
                depth=depth,
                h1_size=spsize,
                h2_size=tpsize,
                partitions=partitions,
            )
            return model(x)

        self.flow = hk.transform(flow_fn)

    def tearDown(self) -> None:
        """TearDown"""
        del self.flow

    def dummy_network_pretrain(self) -> None:
        """Test network pretrain"""
        key = jax.random.PRNGKey(42)
        iterations = 10000
        tolerance_decimal = 1
        tolerance = 1.5 * 10 ** (-tolerance_decimal)
        tolerance = 0.3
        pretrained_params = network_pretrain(
            key=key,
            flow=self.flow,
            x_init=self.x_init,
            z_target=self.z_target,
            iterations=iterations,
            tolerance=tolerance,
        )

        z_transed = self.flow.apply(pretrained_params, None, self.x_init)
        deviance = jnp.linalg.norm(z_transed - self.z_target)
        np.testing.assert_almost_equal(deviance, 0, decimal=tolerance_decimal)


if __name__ == "__main__":
    unittest.main()
    # test = TestPreviousNetworkPretrain()
    # test.dummy_network_pretrain()

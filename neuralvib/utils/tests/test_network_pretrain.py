"""Test network_pretrain.py"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from jax.flatten_util import ravel_pytree

# from neuralvib.molecule.utils.init_molecule import InitMolecule
from neuralvib.networks.flow_MoleNet import MoleNetFlow
from neuralvib.utils.network_pretrain import FlowPretrain
from neuralvib.wfbasis.basis import InvariantGaussian
from neuralvib.wfbasis.basis import InvariantHermiteFunction
from neuralvib.wfbasis.wf_ansatze import WFAnsatz


jax.config.update("jax_enable_x64", True)


class DummyInitMolecule(object):
    """Dummy InitMolecule for test"""

    def __init__(self) -> None:
        """Init"""
        self.molecule = "test"
        self.particles = ("H",)
        self.equivariant_partitions = [0]
        self.particle_mass = 1836.152673
        self.omega = 1 / self.particle_mass


class TestFlowPretrain(unittest.TestCase):
    """Test Flow Pretrain"""

    def setUp(self) -> None:
        """SetUp"""
        self.key = jax.random.PRNGKey(42)
        self.data_path = "./neuralvib/utils/tests/fig"

        print("\n========== Initialize Molecule ==========")
        self.dummy_init_molecule = DummyInitMolecule()
        self.excitation_number = np.array([0] * 3)

        print("\n========== Initialize flow model ==========")
        self.key, subkey = jax.random.split(self.key, 2)
        depth = 3
        spsize = 128
        tpsize = 16

        def flow_fn(x):
            model = MoleNetFlow(
                depth=depth,
                h1_size=spsize,
                h2_size=tpsize,
                partitions=self.dummy_init_molecule.equivariant_partitions,
            )
            return model(x)

        self.flow = hk.transform(flow_fn)
        self.params_flow = self.flow.init(subkey, jnp.zeros((1, 3)))
        raveled_params_flow, _ = ravel_pytree(self.params_flow)
        print(f"\tparameters in the flow model:{raveled_params_flow.size}", flush=True)

        print("\n========== Initialize Wavefunction ==========")
        invariant_hermite_func_obj = InvariantHermiteFunction(
            particles=self.dummy_init_molecule.particles,
            m=self.dummy_init_molecule.particle_mass,
            w=self.dummy_init_molecule.omega,
        )
        wf_ansatze_obj = WFAnsatz(
            flow=self.flow, log_phi_base=invariant_hermite_func_obj.log_phi_base
        )
        self.log_wf_ansatze = wf_ansatze_obj.log_wf_ansatz

    def tearDown(self) -> None:
        """TearDown"""

    def test_network_pretrain(self) -> None:
        """Test network pretrain"""
        key, subkey = jax.random.split(self.key, 2)
        flow_pretrain_obj = FlowPretrain(
            molecule_init_obj=self.dummy_init_molecule,
            key=subkey,
            log_wf_ansatze=self.log_wf_ansatze,
            excitation_number=self.excitation_number,
            init_params=self.params_flow,
            iterations=1000,
            tolerance=1e-3,
            data_path=self.data_path,
            regularization=False,
        )
        params_flow = flow_pretrain_obj.pretrain()


if __name__ == "__main__":
    unittest.main()
    # test = TestFlowPretrain()
    # test.dummy_network_pretrain()

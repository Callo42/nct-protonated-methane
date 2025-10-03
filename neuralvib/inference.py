"""Inference"""

from typing import Callable
import logging
import sys
import argparse
import subprocess

import jax
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from optax._src import base as optax_base

from neuralvib.molecule.utils.init_molecule import InitMolecule
from neuralvib.networks.utils import setting_network
from neuralvib.networks import utils as network_utils
from neuralvib.utils import convert
from neuralvib.utils import ckeckpoint as ckpt_utils
from neuralvib.utils.energy_estimator import EnergyEstimator
from neuralvib.utils.loss import Loss
from neuralvib.utils.mcmc import Metropolis, mcmc_pmap
from neuralvib.utils.update import naive_fori_loop
from neuralvib.wfbasis.basis import HermiteFunction
from neuralvib.wfbasis.wf_ansatze import WFAnsatz


jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Process all messages from DEBUG upwards
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Console shows INFO and above
# file_handler = logging.FileHandler('app.log', mode='a') # 'a' for append
# file_handler.setLevel(logging.DEBUG) # File captures DEBUG and above
# formatter = logging.Formatter(
#     "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
# )
formatter = logging.Formatter("---%(levelname)s---\n%(message)s")
console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
# Avoid adding duplicate handlers if this code might run multiple times
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)


def _set_args() -> argparse.Namespace:
    """Get input args"""
    parser = argparse.ArgumentParser(description="calculate specific state(s) energy.")
    parser.add_argument("--file_name", default="", help="the ckptfile to save data")
    parser.add_argument(
        "--acc_steps",
        default=1,
        type=int,
        help="number of accumulation steps"
        "Note this value times the batch size in"
        " training would result in the final total "
        "batch size: Real_batch_size = acc_step(here)"
        " * batch_size(in training)",
    )
    parser.add_argument("--mc_therm", default=1000, type=int, help="Thermal steps")
    parser.add_argument(
        "--mc_steps", default=1000, type=int, help="number of MCMC steps"
    )
    parser.add_argument(
        "--mc_stddev", default=0.2, type=float, help="standard deviation of MCMC"
    )

    args = parser.parse_args()
    return args


class Inference:
    """Inference"""

    def __init__(
        self,
        mcmc_steps: int,
        batch_size: int,
        num_orb: int,
        excitation_numbers: np.ndarray | jax.Array,
        pmapped_energies_func: Callable,
        acc_steps: int,
        fori_loop_init: dict,
        log_wf_ansatz: Callable,
        batch_info: dict,
        metropolis_sampler_batched: Callable,
        training_args: argparse.Namespace,
    ) -> None:
        self.mcmc_steps = mcmc_steps
        self.batch_size = batch_size
        self.num_orb = num_orb
        self.excitation_numbers = excitation_numbers
        self.pmapped_energies_func = pmapped_energies_func
        self.acc_steps = acc_steps
        self.energies_init = fori_loop_init["energies_init"]
        self.log_wf_ansatz = log_wf_ansatz
        self.metropolis_sampler_batched = metropolis_sampler_batched
        self.num_devices = batch_info["num_devices"]
        self.batch_per_device = batch_info["batch_per_device"]
        self.training_args = training_args
        logger.debug(f"acc_steps={self.acc_steps}")

    def run(
        self,
        key: jax.Array,
        xs_batched: jax.Array,
        probability_batched: jax.Array,
        mc_step_size: jax.Array,
        params: jax.Array | np.ndarray | dict | optax_base.Params,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """Run the inference
        NOTE: call WITHOUT vmap!

        Args:
            key: the jax.PRNGkey
            xs_batched: (num_of_batch,num_orb, num_of_particles,dim,)
                the batched configuration cartesian coordinate of the particle(s).
            probability_batched: (num_of_batch,num_orb)
                the batched probability for each state (wavefunction**2)
            mc_step_size: (num_orb,) last mcmc moving step size.
                NOTE: this is a per orbital property!
            params: the flow parameters

        Returns:
            energies: (num_orb,) the meaned energies of each orbital
            energies_std: (num_orb,) the standard deviation of each orbital's
                energies.
            kinetics: (num_orb,) the meaned kinetic energies of each orbital.
            kinetics_std:(num_orb,) the standard deviation of each orbital's
                kinetic energies.
            potentials: (num_orb,) the meaned potential energies of each orbital.
            potentials_std: (num_orb,) the standard deviation of each orbital's
                potential energies.
        """

        def _acc_body_func(i, val):
            """Gradient Accumulation Body Function"""
            (
                energies,
                kinetics,
                potentials,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_in,
                params,
            ) = val
            (
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_per_orb,
            ) = mcmc_pmap(
                self.mcmc_steps,
                key,
                xs_batched,
                self.excitation_numbers,
                params,
                probability_batched,
                mc_step_size,
                self.metropolis_sampler_batched,
            )
            loss_i, energies_i, kinetics_i, potentials_i = self.pmapped_energies_func(
                params, xs_batched
            )  # energies:(num_device, batch_per_device, num_orb)
            (energies, kinetics, potentials) = jax.tree.map(
                lambda acc, i: acc + i,
                (energies, kinetics, potentials),
                (energies_i, kinetics_i, potentials_i),
            )
            return (
                energies,
                kinetics,
                potentials,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_per_orb,
                params,
            )

        xs_batched = xs_batched.reshape(
            self.num_devices,
            self.batch_per_device,
            self.num_orb,
            self.training_args.num_of_particles,
            self.training_args.dim,
        )
        probability_batched = probability_batched.reshape(
            self.num_devices, self.batch_per_device, self.num_orb
        )
        energies_init = self.energies_init.reshape(
            self.num_devices, self.batch_per_device, self.num_orb
        )
        key = jax.random.split(key, self.num_devices)

        acc_init_val = (
            jnp.zeros_like(energies_init),
            jnp.zeros_like(energies_init),
            jnp.zeros_like(energies_init),
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            jnp.zeros(self.num_orb, dtype=jnp.float64),
            params,
        )

        (
            energies_batch,
            kinetics_batch,
            potentials_batch,
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            pmove_per_orb,
            params,
        ) = naive_fori_loop(0, self.acc_steps, _acc_body_func, acc_init_val)

        (energies_batch, kinetics_batch, potentials_batch) = jax.tree.map(
            lambda acc: acc / self.acc_steps,
            (energies_batch, kinetics_batch, potentials_batch),
        )

        # (batch,num_orb)
        energies_batch = energies_batch.reshape(
            self.num_devices * self.batch_per_device, self.num_orb
        )
        kinetics_batch = kinetics_batch.reshape(
            self.num_devices * self.batch_per_device, self.num_orb
        )
        potentials_batch = potentials_batch.reshape(
            self.num_devices * self.batch_per_device, self.num_orb
        )
        # (num_orb,)
        energies = jnp.mean(energies_batch, axis=0)
        kinetics = jnp.mean(kinetics_batch, axis=0)
        potentials = jnp.mean(potentials_batch, axis=0)

        energies_std = jnp.sqrt(
            ((energies_batch**2).mean(axis=0) - energies**2)
            / (self.acc_steps * self.batch_size)
        )
        kinetics_std = jnp.sqrt(
            ((kinetics_batch**2).mean(axis=0) - kinetics**2)
            / (self.acc_steps * self.batch_size)
        )
        potentials_std = jnp.sqrt(
            ((potentials_batch**2).mean(axis=0) - potentials**2)
            / (self.acc_steps * self.batch_size)
        )

        return (
            energies,
            energies_std,
            kinetics,
            kinetics_std,
            potentials,
            potentials_std,
        )


def main():
    """The main function for getting energy"""
    print("jax.__version__:", jax.__version__)
    print(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True
        ).stdout,
        flush=True,
    )

    args = _set_args()
    logger.info(f"Loading file: {args.file_name}")
    ckptfile = ckpt_utils.load_data(args.file_name)
    training_args = ckptfile["training_args"]
    key = ckptfile["key"]
    x = ckptfile["x"]
    params_flow = ckptfile["params_flow"]

    logger.info("========== Checking Num_Of_Devices ==========")
    num_devices = jax.device_count()
    if training_args.batch % num_devices != 0:
        logger.error("XLA Device number mismatch!")
        raise ValueError(
            "Batch size must be dividible by the number of GPU devices. "
            f"Got batch = {training_args.batch} for {num_devices} devices now."
        )
    batch_per_device = training_args.batch // num_devices

    key, subkey = jax.random.split(key, 2)
    convert_hartree_to_cm_inv_coefficient = convert.convert_hartree_to_inverse_cm(1.0)

    logger.info("==================== Init Molecule =======================")
    logger.info(
        f"Molecule:{training_args.molecule}\n"
        f"number of particles:{training_args.num_of_particles}"
    )

    molecule_init_obj = InitMolecule(
        molecule=training_args.molecule,
        x_alpha=training_args.x_alpha,
        input_args=training_args,
    )
    # pes_func: (num_of_particles,dim,)->(1,)
    pes_cartesian = molecule_init_obj.pes_cartesian

    logger.info("========== Initialize Excitation Number ==========")
    # Shape: (num_of_orb, num_of_particles * dim)
    excitation_numbers = molecule_init_obj.excitation_numbers(
        num_of_orb=training_args.num_orb
    )

    logger.info("========== Initialize flow model ==========")
    network_config = setting_network(input_args=training_args)
    key, subkey = jax.random.split(key, 2)
    flow = network_utils.make_flow(
        network_config=network_config,
        key=subkey,
        input_args=training_args,
        molecule_init_obj=molecule_init_obj,
    )
    logger.info("Network Config:\n")
    for temp1, temp2 in network_config.items():
        logger.info(f"{temp1}:{temp2}")
    key, subkey = jax.random.split(key, 2)
    raveled_params_flow, _ = ravel_pytree(params_flow)
    logger.info(f"\tparameters in the flow model:{raveled_params_flow.size}")

    logger.info(
        "================ Initializing coordinates as sigma=1 normal ============="
    )
    # logger.info("================ Loading coordinates =============")
    # x = x.reshape(
    #     training_args.batch,
    #     training_args.num_orb,
    #     training_args.num_of_particles,
    #     training_args.dim,
    # )
    key, subkey = jax.random.split(key)
    x = 1.0 * jax.random.normal(
        subkey,
        shape=(
            training_args.batch,
            training_args.num_orb,
            training_args.num_of_particles,
            training_args.dim,
        ),
    )
    logger.info(f"x.shape={x.shape}")

    logger.info("========== Initialize Wavefunction ==========")
    hermite_func_obj = HermiteFunction(
        molecule_init_obj,
    )
    wf_ansatze_obj = WFAnsatz(
        flow=flow,
        log_phi_base=hermite_func_obj.log_phi_base,
        training_args=training_args,
    )
    log_wf_ansatze = wf_ansatze_obj.log_wf_ansatz

    logger.info("========== Initialize Metropolis ==========")
    metropolis = Metropolis(
        wf_ansatz=log_wf_ansatze,
        particles=molecule_init_obj.particles,
        particle_mass=molecule_init_obj.particle_mass,
    )
    metropolis_oneshot_sample = metropolis.oneshot_sample
    metropolis_sample_batched = jax.vmap(  # vmap on batch
        jax.vmap(  # vmap on num_orb
            metropolis_oneshot_sample, in_axes=(0, 0, 0, None, 0, 0)
        ),
        in_axes=(0, None, 0, None, None, 0),
    )
    probability_batched = (
        jax.vmap(
            jax.vmap(log_wf_ansatze, in_axes=(None, 0, 0)), in_axes=(None, 0, None)
        )(params_flow, x, excitation_numbers)
        * 2
    )
    step_size = args.mc_stddev * np.ones(training_args.num_orb)

    logger.info("========== Thermalize ==========")
    subkey = jax.random.split(key, num_devices)
    for ii in range(args.mc_therm):
        x = x.reshape(
            num_devices,
            batch_per_device,
            training_args.num_orb,
            training_args.num_of_particles,
            training_args.dim,
        )
        probability_batched = probability_batched.reshape(
            num_devices,
            batch_per_device,
            training_args.num_orb,
        )
        key, x, probability_batched, step_size, pmove_per_orb = mcmc_pmap(
            training_args.mc_steps,
            subkey,
            x,
            excitation_numbers,
            params_flow,
            probability_batched,
            step_size,
            metropolis_sample_batched,
        )

        # step_size = jnp.where(
        #     pmove_per_orb > 0.55, step_size * 1.05, step_size
        # )
        # step_size = jnp.where(
        #     pmove_per_orb < 0.30, step_size * 0.905, step_size
        # )
        subkey = jax.random.split(key[0], num_devices)
    print(
        f"---- After Thermalization,  ac(pmove): {pmove_per_orb},",
        f"  stepsize: {step_size},  ----",
        flush=True,
    )
    x = x.reshape(
        training_args.batch,
        training_args.num_orb,
        training_args.num_of_particles,
        training_args.dim,
    )
    probability_batched = probability_batched.reshape(
        training_args.batch, training_args.num_orb
    )

    key = key[0]

    logger.info("========== Initialize Energy Estimator ==========")
    energy_estimator = EnergyEstimator(
        wf_ansatz=log_wf_ansatze,
        potential_func=pes_cartesian,
        particles=molecule_init_obj.particles,
        particle_mass=molecule_init_obj.particle_mass,
        pot_batch_method=molecule_init_obj.pot_parallel_method,
        no_kinetic=training_args.no_kinetic,
    )

    logger.info("========== Initialize Loss Object ==========")
    loss_obj = Loss(
        wf_ansatz=log_wf_ansatze,
        batched_local_energy_estimator=energy_estimator.batched_local_energy,
        excitation_numbers=excitation_numbers,
        clip_factor=training_args.clip_factor,
    )
    pmapped_total_energies = jax.pmap(
        loss_obj.total_energy,
        axis_name="xla_device",
        in_axes=(None, 0),
        out_axes=(None, 0, 0, 0),
    )
    for_i_loop_init = {
        "energies_init": jnp.zeros((training_args.batch, training_args.num_orb)),
    }

    logger.info("========== Initialize Inference Object ==========")
    key, subkey = jax.random.split(key, 2)
    inference_kernel = Inference(
        mcmc_steps=args.mc_steps,
        batch_size=training_args.batch,
        num_orb=training_args.num_orb,
        excitation_numbers=excitation_numbers,
        pmapped_energies_func=pmapped_total_energies,
        acc_steps=args.acc_steps,
        fori_loop_init=for_i_loop_init,
        log_wf_ansatz=log_wf_ansatze,
        batch_info={"num_devices": num_devices, "batch_per_device": batch_per_device},
        metropolis_sampler_batched=metropolis_sample_batched,
        training_args=training_args,
    )

    logger.info("========== Running Inference ==========")
    energies, energies_std, _, _, _, _ = inference_kernel.run(
        key=subkey,
        xs_batched=x,
        probability_batched=probability_batched,
        mc_step_size=step_size,
        params=params_flow,
    )
    energies = np.array(energies * convert_hartree_to_cm_inv_coefficient)
    energies_std = np.array(energies_std * convert_hartree_to_cm_inv_coefficient)

    logger.info("========== Energies (Sorted) ==========")
    energies_sort_index = np.argsort(energies)
    energies_sorted = energies[energies_sort_index]
    energies_std_sorted = energies_std[energies_sort_index]
    energies_difference = energies_sorted - energies_sorted[0]  # relative energy
    logger.info(f"sorting index={energies_sort_index}")
    logger.info(f"energies_sorted=\n{energies_sorted}")
    logger.info(f"energies_difference=\n{energies_difference}")
    logger.info(f"energies_std_sorted=\n{energies_std_sorted}")


if __name__ == "__main__":
    main()

"""Train"""

import os
import time
import subprocess
import argparse
import warnings
from datetime import datetime
import json

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

from neuralvib.molecule.utils.init_molecule import InitMolecule
from neuralvib.networks.utils import setting_network
from neuralvib.networks import utils as network_utils
from neuralvib.utils import convert
from neuralvib.utils import ckeckpoint as ckpt_utils
from neuralvib.utils.energy_estimator import EnergyEstimator
from neuralvib.utils.initialize import init_batched_x
from neuralvib.utils.loss import Loss

from neuralvib.utils.mcmc import Metropolis, mcmc_pmap
from neuralvib.utils.update import Update
from neuralvib.utils.network_pretrain import FlowPretrain

# from neuralvib.wfbasis.basis import InvariantGaussian
from neuralvib.wfbasis.basis import HermiteFunction
from neuralvib.wfbasis.wf_ansatze import WFAnsatz
from neuralvib.utils.boltzmann_fac import (
    boltzmann_probabilities_au,
    k_lowest_harmonic_levels_au,
)

jax.config.update("jax_enable_x64", True)


def adjust_mcmc_step_size(
    step_size, pmove_per_orb, upper_thr, lower_thr, inc_fac, dec_fac
):
    """Adjust MCMC Gaussian proposal step sizes based on per-orbital move acceptance."""
    # pmove_per_orb: (num_orb,)
    step_size = jnp.where(pmove_per_orb > upper_thr, step_size * inc_fac, step_size)
    step_size = jnp.where(pmove_per_orb < lower_thr, step_size * dec_fac, step_size)
    return step_size


def set_args():
    """Setting input args."""
    parser = argparse.ArgumentParser(
        description="simulation for the harmonic oscillator"
    )

    # folder to save data.
    parser.add_argument(
        "--folder",
        default="./data/",
        help="the folder to save data",
    )

    # pretrain model ckpt file
    parser.add_argument(
        "--ckpt_file_folder", default=None, help="The ckpt file's folder."
    )

    # physical parameters.
    parser.add_argument(
        "--molecule",
        type=str,
        default="CH5+",
        choices=["CH5+", "CH4"],
        help="molecule to compute, include toy models."
        "Note: the option User is for user specified potential files."
        "CH5+: the CH5+ molecule,"
        "CH4: the CH4 molecule.",
    )
    parser.add_argument(
        "--num_of_particles", type=int, default=3, help="total number of particles"
    )
    parser.add_argument(
        "--dim", type=int, default=3, help="spatial dimension of particles"
    )
    parser.add_argument(
        "--select_potential",
        type=str,
        default=None,
        help="The potential energy surface to use."
        "Avaliable: jax-original, jax-callback and external-pes."
        "1. jax-original: for example, the CH4 PES `J.Chem.Phys.102,254-261(1995)`"
        "refers to the jax implemented PES; 2. jax-callback: for example, the "
        "CH5+ JBB Full PES, `J.Phys.Chem.A2006,110,1569-1574`"
        "or `PySCF.MP2.pure` or `PySCF.HF.io` refers to the"
        "PES that is originally implemented in non-jax code and using callback."
        "3. external-pes: MUST desinate `external`,"
        "if designated, means directly use external batched PES call,"
        "for example, `External.PySCF.DFT.Joblib` stands for parallel "
        "implemented by joblib,"
        "rather than using jax.vmap to compute batched configurations.",
    )
    parser.add_argument(
        "--x_alpha",
        type=float,
        default=1.0,
        help="the scaled_x scaling factor for training. Typically set to 1 is ok.",
    )
    parser.add_argument(
        "--init_ref_noise",
        type=float,
        default=1e-2,
        help=(
            "Stddev of Gaussian noise added to the equilibrium geometry before "
            "fixing to the Eckart frame when initializing walkers (in a0). "
            "Set to 0.0 for deterministic initialization."
        ),
    )

    parser.add_argument("--num_orb", type=int, default=1, help="number of orbitals")
    parser.add_argument(
        "--choose_orb", type=int, nargs="+", default=None, help="choose orbital"
    )

    # flow model: neural autoregressive flow & deep dense sigmoid flow parameters.
    parser.add_argument(
        "--flow_type",
        type=str,
        default="MoleNet",
        choices=["MoleNet", "RNVP"],
        help="The flow type. MoleNet is the equivariant flow."
        " RNVP is the Real-valued Non-Volume-Preserving flow.",
    )
    parser.add_argument("--flow_depth", type=int, default=3, help="Flow depth")
    parser.add_argument(
        "--mlp_width",
        type=int,
        default=16,
        help="NAF/Real NVP: width of the hidden layers",
    )
    parser.add_argument(
        "--mlp_depth",
        type=int,
        default=2,
        help="NAF/Real NVP: depth of the hidden layers",
    )
    parser.add_argument(
        "--molenet_spsize",
        type=int,
        default=32,
        help="MoleNet single particle stream hidden units",
    )
    parser.add_argument(
        "--molenet_tpsize",
        type=int,
        default=16,
        help="MoleNet two particle stream hidden units",
    )
    parser.add_argument(
        "--molenet_init_stddev",
        type=float,
        default=0.0001,
        help="MoleNet init_stddev",
    )

    # training parameters.
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="batch size (per single gradient accumulation step)",
    )
    parser.add_argument(
        "--acc_steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--epoch_finished",
        type=int,
        default=0,
        help="number of epochs already finished.",
    )
    parser.add_argument("--epoch", type=int, default=2000, help="final epoch")
    parser.add_argument(
        "--no-kinetic",
        action="store_true",
        help=(
            "if designated, then set K=0 in training,"
            "this could refer to the program only optimizing "
            "grad_theta lnp(E(x)-E_average), and resulting in "
            "seeking the minimum of the PES. "
            "Note this could also be used as the pretrain of the flow."
        ),
    )
    parser.add_argument(
        "--pretrain-network", action="store_true", help="pretrian the network"
    )
    parser.add_argument(
        "--pretrain-batch",
        type=int,
        default=50,
        help="pretrain batch size",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=1,
        help="number of epochs for pretraining the flow model",
    )

    # optimizer parameters.
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help=(
            "optimizer type."
            "Note that sr_cg is for using Conjugate Gradient iteration"
            "(CG) algorithm to solve Ax=b inside a sr optimizer."
        ),
    )
    # Adam
    parser.add_argument(
        "--adam_lr",
        type=float,
        default=1e-3,
        help="learning rate (for adam and adamw)",
    )
    # Adamw
    parser.add_argument(
        "--adamw_decay", type=float, default=None, help="weight_decay for adamw"
    )

    # mcmc
    parser.add_argument(
        "--mc_therm", type=int, default=20, help="MCMC thermalization steps"
    )
    parser.add_argument("--mc_steps", type=int, default=100, help="MCMC update steps")
    parser.add_argument(
        "--mc_stddev",
        type=float,
        default=0.5,
        help="standard deviation of the Gaussian proposal in MCMC update",
    )
    parser.add_argument(
        "--mc_selfadjust_stepsize",
        action="store_true",
        help="if designated, then self adjust the stepsize after MCMC update",
    )
    parser.add_argument(
        "--mc_adjust_upper",
        type=float,
        default=0.65,
        help="Acceptance upper threshold to increase step size",
    )
    parser.add_argument(
        "--mc_adjust_lower",
        type=float,
        default=0.25,
        help="Acceptance lower threshold to decrease step size",
    )
    parser.add_argument(
        "--mc_adjust_increase_factor",
        type=float,
        default=1.05,
        help="Multiplicative factor when acceptance > upper threshold",
    )
    parser.add_argument(
        "--mc_adjust_decrease_factor",
        type=float,
        default=0.95,
        help="Multiplicative factor when acceptance < lower threshold",
    )
    parser.add_argument(
        "--clip_factor", type=float, default=5.0, help="clip factor for gradient"
    )
    parser.add_argument(
        "--boltzmann_weight_T",
        type=float,
        default=10000.0,
        help="Boltzmann weight temperature (in K)",
    )

    parser.add_argument(
        "--ckpt_epochs", type=int, default=100, help="save checkpoint every ckpt_epochs"
    )
    parser.add_argument(
        "--excite_gen_type",
        type=int,
        required=True,
        help="Excitation generation type: 1, 2, or 3."
        "corresponding to InitMolecule.generate_combinations"
        ", InitMolecule.generate_combinations_2, or energy-prioritized"
        " InitMolecule.generate_combinations_3",
    )

    # ========== args.params -> params ==========
    args = parser.parse_args()
    return args


def training_kernel() -> None:
    """Training Kernel"""
    # Get the current date and time
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d/%H-%M-%S")

    print("jax.__version__:", jax.__version__)
    try:
        print(
            subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True
            ).stdout,
            flush=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("nvidia-smi not available; running on CPU.", flush=True)

    training_args = set_args()
    network_config = setting_network(input_args=training_args)
    key = jax.random.PRNGKey(42)

    convert_hartree_to_cm_inv_coefficient = convert.convert_hartree_to_inverse_cm(1.0)

    print("\n========== Initialize Molecule ==========")
    print(
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
    mole_instance = molecule_init_obj.mole_instance
    w_indices = mole_instance.w_indices
    levels_for_boltzmann = k_lowest_harmonic_levels_au(
        w_indices, max_states=training_args.num_orb
    )
    boltzmann_factor = boltzmann_probabilities_au(
        levels_for_boltzmann, temperature_K=training_args.boltzmann_weight_T
    )
    ms = mole_instance.ms
    print(f"ms array:\n{ms}")
    print(f"\nBoltzmann weights at T={training_args.boltzmann_weight_T} K:")
    print(boltzmann_factor)

    print("\n========== Initialize Excitation Number ==========")
    # Shape: (num_of_orb, num_of_particles * dim)
    excitation_numbers = molecule_init_obj.excitation_numbers(
        num_of_orb=training_args.num_orb
    )
    excitation_numbers = jax.lax.stop_gradient(
        jnp.array(excitation_numbers, dtype=jnp.int32)
    )

    print("\n========== Initialize flow model ==========")
    key, subkey = jax.random.split(key, 2)
    flow = network_utils.make_flow(
        network_config=network_config,
        key=subkey,
        input_args=training_args,
        molecule_init_obj=molecule_init_obj,
    )
    print("Network Config:\n")
    for temp1, temp2 in network_config.items():
        print(f"{temp1}:{temp2}")
    key, subkey = jax.random.split(key, 2)
    params_flow = flow.init(
        subkey, jnp.zeros((training_args.num_of_particles, training_args.dim))
    )
    raveled_params_flow, _ = ravel_pytree(params_flow)
    print(f"\tparameters in the flow model:{raveled_params_flow.size}", flush=True)

    print("\n================ Initialize coordinates =============", flush=True)
    key, subkey = jax.random.split(key, 2)
    x = init_batched_x(
        key=subkey,
        batch_size=training_args.batch,
        num_orb=training_args.num_orb,
        num_of_particles=training_args.num_of_particles,
        dim=training_args.dim,
        init_ref_noise=training_args.init_ref_noise,
        molecule=training_args.molecule,
        ms=ms,
        x_ref=mole_instance.equilibrium_config,
    )
    print("x.shape:", x.shape)

    print("\n========== Initialize Wavefunction ==========")
    hermite_func_obj = HermiteFunction(
        molecule_init_obj,
        sphere_radius=None,
    )
    wf_ansatze_obj = WFAnsatz(
        flow=flow,
        log_phi_base=hermite_func_obj.log_phi_base,
        training_args=training_args,
        ms=ms,
        x_ref=mole_instance.equilibrium_config,
    )
    log_wf_ansatze = wf_ansatze_obj.log_wf_ansatz

    print("\n========== Initialize Metropolis ==========")
    metropolis = Metropolis(
        wf_ansatz=log_wf_ansatze,
        ms=ms,
        x_ref=mole_instance.equilibrium_config,
    )
    metropolis_oneshot_sample = metropolis.oneshot_sample
    metropolis_sample_batched = jax.vmap(  # vmap on batch
        jax.vmap(  # vmap on num_orb
            metropolis_oneshot_sample, in_axes=(0, 0, 0, None, 0, 0)
        ),
        in_axes=(0, None, 0, None, None, 0),
    )

    print("\n========== Initialize Energy Estimator ==========")
    energy_estimator = EnergyEstimator(
        wf_ansatz=log_wf_ansatze,
        potential_func=pes_cartesian,
        particles=molecule_init_obj.particles,
        particle_mass=molecule_init_obj.particle_mass,
        pot_batch_method=molecule_init_obj.pot_parallel_method,
        no_kinetic=training_args.no_kinetic,
    )

    print("\n========== Initialize optimizer ==========")
    pct_start = 0.3
    div_fac = 25
    final_div_fac = 1e4
    scheduler = optax.cosine_onecycle_schedule(
        transition_steps=training_args.epoch,
        peak_value=training_args.adam_lr,
        pct_start=pct_start,
        div_factor=div_fac,
        final_div_factor=final_div_fac,
    )
    print(
        f"LR scheduler: OneCycle (cosine_onecycle_schedule)\n"
        f"    peak_value      = {training_args.adam_lr}\n"
        f"    pct_start      = {pct_start}\n"
        f"    div_factor      = {div_fac}\n"
        f"    final_div_factor      = {final_div_fac}\n"
    )
    if training_args.optimizer == "adam":
        optimizer = optax.adam(learning_rate=scheduler)
        print(f"Optimizer adam:\n    learning rate: {training_args.adam_lr}")
    else:
        raise ValueError(
        f"Unsupported optimizer '{training_args.optimizer}'. "
        "This implementation is only validated with Adam; "
        "using other optimizers may lead to unstable or incorrect training. "
        "Please switch to 'adam' or update the training code to handle other optimizers explicitly."
    )

    print("\n========== Checking Num_Of_Devices ==========")
    num_devices = jax.device_count()
    if training_args.batch % num_devices != 0:
        raise ValueError(
            "Batch size must be divisible by the number of GPU devices. "
            f"Got batch = {training_args.batch} for {num_devices} devices now."
        )
    batch_per_device = training_args.batch // num_devices

    print("\n========== Check point ==========")
    if training_args.epoch_finished != 0:  # try to load from ckptfile
        raise NotImplementedError("Loading from checkpoint is not rechecked yet.")
        if training_args.pretrain_network:
            raise ValueError(
                "pretrain_network is not supported when starting from ckpt."
            )
        path = training_args.ckpt_file_folder
        # Loading target checkpoint file
        load_ckpt_filename = ckpt_utils.ckpt_filename(
            training_args.epoch_finished, training_args.ckpt_file_folder
        )
        if not os.path.isfile(load_ckpt_filename):
            raise FileNotFoundError(
                "Checkpoint file not found"
                f" while designating epoch_finished={training_args.epoch_finished}\n"
            )

        ckpt = ckpt_utils.load_data(load_ckpt_filename)
        print(f"Load checkpoint file:{load_ckpt_filename}")

        key = ckpt["key"]
        key, subkey = jax.random.split(key, 2)

        (
            opt_state,
            params_flow,
        ) = (
            ckpt["opt_state"],
            ckpt["params_flow"],
        )
        probability_batched = (
            jax.vmap(
                jax.vmap(log_wf_ansatze, in_axes=(None, 0, 0)), in_axes=(None, 0, None)
            )(params_flow, x, excitation_numbers)
            * 2
        )

        print("\n========== Thermalize ==========")
        step_size = training_args.mc_stddev * np.ones(training_args.num_orb)
        subkey = jax.random.split(key, num_devices)
        for ii in range(training_args.mc_therm):
            t0 = time.time()
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
            t1 = time.time()
            print(
                f"---- thermal step: {ii+1},  ac(pmove): {pmove_per_orb},",
                f"  stepsize: {step_size},  time: {(t1-t0):.4f} ----",
                flush=True,
            )
            subkey = jax.random.split(key[0], num_devices)
            if training_args.mc_selfadjust_stepsize:
                print("Self adjust step size after each mc step ")
                step_size = adjust_mcmc_step_size(
                    step_size,
                    pmove_per_orb,
                    training_args.mc_adjust_upper,
                    training_args.mc_adjust_lower,
                    training_args.mc_adjust_increase_factor,
                    training_args.mc_adjust_decrease_factor,
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
    else:
        print("Epoch_finished==0. Start from scratch.")

        print("\n========== Init Data Directory ==========")
        # ========== create file path ==========
        molecule_str = training_args.molecule + (f"_n_{training_args.num_of_particles}")

        flow_str = network_config["flow_string"]

        if training_args.optimizer == "adam":
            optimizer_str = f"_adam_lr_{training_args.adam_lr}"
        elif training_args.optimizer == "adamw":
            optimizer_str = (
                f"_adamw_lr_{training_args.adam_lr}_"
                f"wdcy_{training_args.adamw_weight_decay}"
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_args.optimizer}")

        path = training_args.folder
        if training_args.pretrain_network:
            path = os.path.join(
                path,
                f"PretrainBatch_{training_args.pretrain_batch}",
            )
        path = os.path.join(
            path,
            (
                molecule_str
                + (f"_orb_{training_args.num_orb}")
                + flow_str
                + optimizer_str
                + (f"_bth_{training_args.batch}_acc_{training_args.acc_steps}")
                + (
                    f"_mcthr_{training_args.mc_therm}_stp_{training_args.mc_steps}"
                    f"_std_{training_args.mc_stddev}"
                )
                + (f"_clp_{training_args.clip_factor}")
            ),
        )
        path = os.path.join(path, formatted_datetime)

        print("#file path:", path)
        if not os.path.isdir(path):
            os.makedirs(path)
            print(f"#create path: {path}")

        if training_args.pretrain_network:
            raise ValueError(
                "Since we obtained good behaviour under CoM, Pretrain is disabled by default."
            )
            print("\n========== Pretraining flow model ==========")
            print(f"Pretrain batch size: {training_args.pretrain_batch}")
            print(f"Pretrain epochs: {training_args.pretrain_epochs}")
            key, subkey = jax.random.split(key, 2)
            flow_pretain_obj = FlowPretrain(
                molecule_init_obj=molecule_init_obj,
                key=subkey,
                log_wf_ansatze=log_wf_ansatze,
                excitation_number=excitation_numbers[0],
                init_params=params_flow,
                iterations=training_args.pretrain_epochs,
                pretrain_batch=training_args.pretrain_batch,
                data_path=path,
                tolerance=0.50,
            )
            params_flow = flow_pretain_obj.pretrain()
            del flow_pretain_obj
        # ========== optimizer state ==========
        print("initialize optimizer state...", flush=True)
        if training_args.optimizer == "adam":
            opt_state = optimizer.init(params_flow)
        elif training_args.optimizer == "adamw":
            opt_state = optimizer.init(params_flow)

        print("\n========== Thermalize ==========")
        step_size = training_args.mc_stddev * np.ones(training_args.num_orb)
        probability_batched = (
            jax.vmap(
                jax.vmap(log_wf_ansatze, in_axes=(None, 0, 0)),
                in_axes=(None, 0, None),
            )(params_flow, x, excitation_numbers)
            * 2
        )

        subkey = jax.random.split(key, num_devices)
        for ii in range(training_args.mc_therm):
            t0 = time.time()
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
            t1 = time.time()
            print(
                f"---- thermal step: {ii+1},  ac(pmove): {pmove_per_orb},",
                f"  stepsize: {step_size},  time: {(t1-t0):.4f} ----",
                flush=True,
            )
            if training_args.mc_selfadjust_stepsize:
                print("Self adjust step size after each mc step ")
                step_size = adjust_mcmc_step_size(
                    step_size,
                    pmove_per_orb,
                    training_args.mc_adjust_upper,
                    training_args.mc_adjust_lower,
                    training_args.mc_adjust_increase_factor,
                    training_args.mc_adjust_decrease_factor,
                )
            subkey = jax.random.split(key[0], num_devices)
        x = x.reshape(
            training_args.batch,
            training_args.num_orb,
            training_args.num_of_particles,
            training_args.dim,
        )
        probability_batched = probability_batched.reshape(
            training_args.batch, training_args.num_orb
        )

    # If `key` is sharded across devices (shape: [num_devices, 2]),
    # collapse back to a single host key; otherwise leave as-is.
    if key.ndim > 1:
        key = key[0]

    print("\n========== Initialize Loss Object ==========")
    x = x.reshape(
        num_devices,
        batch_per_device,
        training_args.num_orb,
        training_args.num_of_particles,
        training_args.dim,
    )

    loss_obj = Loss(
        wf_ansatz=log_wf_ansatze,
        batched_local_energy_estimator=energy_estimator.batched_local_energy,
        excitation_numbers=excitation_numbers,
        clip_factor=training_args.clip_factor,
        boltzmann_weights=boltzmann_factor,
    )
    loss_and_grad = jax.pmap(
        loss_obj.loss_and_grad_pmap,
        axis_name="xla_device",
        in_axes=(None, 0),
        out_axes=(None, 0, 0, 0, None),
    )
    loss, energies, kinetics, potentials, gradients = loss_and_grad(params_flow, x)

    x = x.reshape(
        training_args.batch,
        training_args.num_orb,
        training_args.num_of_particles,
        training_args.dim,
    )
    energies = energies.reshape(training_args.batch, training_args.num_orb)
    kinetics = kinetics.reshape(training_args.batch, training_args.num_orb)
    potentials = potentials.reshape(training_args.batch, training_args.num_orb)

    print(
        "After MCMC, with initial network,"
        f" energy={loss*convert_hartree_to_cm_inv_coefficient:.2f}"
    )
    print(
        "Average Kinetics="
        f"{jnp.mean(kinetics, axis=0)*convert_hartree_to_cm_inv_coefficient};\n"
        "Average potentials="
        f"{jnp.mean(potentials, axis=0)*convert_hartree_to_cm_inv_coefficient}"
    )

    if training_args.epoch_finished == 0:
        print("\n========== Save Checkpoint Before Training ===========")
        if training_args.optimizer == "adam":
            opt_state_ckpt = opt_state
        elif training_args.optimizer == "adamw":
            opt_state_ckpt = opt_state
        ckpt = {
            "training_args": training_args,
            "key": key,
            "x": x,
            "opt_state": opt_state_ckpt,
            "params_flow": params_flow,
            "boltzmann_weights": boltzmann_factor,
        }
        save_ckpt_filename = ckpt_utils.ckpt_filename(0, path)
        ckpt_utils.save_data(ckpt, save_ckpt_filename)
        print(f"save file: {save_ckpt_filename}", flush=True)

    print("\n========== Initialize Update Object ==========")
    batch_info = {
        "num_devices": num_devices,
        "batch_per_device": batch_per_device,
    }
    fori_loop_init = {
        "loss_init": loss,
        "energies_init": energies,
        "grad_init": gradients,
    }
    update_obj = Update(
        mcmc_steps=training_args.mc_steps,
        batch_size=training_args.batch,
        num_orb=training_args.num_orb,
        excitation_numbers=excitation_numbers,
        loss_and_grad=loss_and_grad,
        optimizer=optimizer,
        acc_steps=training_args.acc_steps,
        fori_loop_init=fori_loop_init,
        log_wf_ansatz=log_wf_ansatze,
        metropolis_sampler_batched=metropolis_sample_batched,
        batch_info=batch_info,
        training_args=training_args,
        boltzmann_weights=boltzmann_factor,
    )
    update_func = update_obj.update

    # Save training_args to JSON in the same directory as data
    try:
        args_json_path = os.path.join(path, "training_args.json")
        if not os.path.isfile(args_json_path):
            with open(args_json_path, "w", encoding="utf-8") as jf:
                json.dump(vars(training_args), jf, indent=2)
            print(f"Saved training args: {args_json_path}")
        else:
            print(f"training_args.json already exists at: {args_json_path}")
    except Exception as e:
        warnings.warn(f"Failed to save training_args.json: {e}")

    print("\n========== Open Data File ==========")
    log_filename = os.path.join(path, "data.txt")
    print("#data name: ", log_filename, flush=True)
    with open(
        log_filename,
        ("w" if training_args.epoch_finished == 0 else "a"),
        buffering=1,
        newline="\n",
        encoding="utf-8",
    ) as f:
        print("\n========== Start Training ==========")
        t0 = time.time()
        for ii in range(training_args.epoch_finished + 1, training_args.epoch + 1):
            t1 = time.time()
            key, subkey = jax.random.split(key, 2)
            (
                loss_energy,
                loss_energy_std,
                kinetic_energy,
                kinetic_energy_std,
                potential_energy,
                potential_energy_std,
                raw_levels_mean,
                x,
                probability_batched,
                params_flow,
                opt_state,
                pmove_per_orb,
                step_size,
            ) = update_func(
                subkey,
                x,
                probability_batched,
                step_size,
                params_flow,
                opt_state,
            )

            epoch_i_energy = loss_energy * training_args.x_alpha
            epoch_i_energy_std = loss_energy_std * training_args.x_alpha
            epoch_i_kinetic = kinetic_energy * training_args.x_alpha
            epoch_i_kinetic_std = kinetic_energy_std * training_args.x_alpha
            epoch_i_potential = potential_energy * training_args.x_alpha
            epoch_i_potential_std = potential_energy_std * training_args.x_alpha

            epoch_i_energy *= convert_hartree_to_cm_inv_coefficient
            epoch_i_energy_std *= convert_hartree_to_cm_inv_coefficient
            epoch_i_kinetic *= convert_hartree_to_cm_inv_coefficient
            epoch_i_kinetic_std *= convert_hartree_to_cm_inv_coefficient
            epoch_i_potential *= convert_hartree_to_cm_inv_coefficient
            epoch_i_potential_std *= convert_hartree_to_cm_inv_coefficient

            t2 = time.time()
            # Compute per-level energy differences to ground (min) in cm^-1
            levels_cm = (
                np.array(update_obj.last_raw_levels_mean)
                * training_args.x_alpha
                * convert_hartree_to_cm_inv_coefficient
            )
            levels_sorted_idx = np.argsort(levels_cm)
            levels_sorted = levels_cm[levels_sorted_idx]
            diffs_sorted = levels_sorted - levels_sorted[0]
            levels_stream_str = ", ".join(
                f"{idx}:{val:.1f}({diff:.1f})"
                for idx, val, diff in zip(
                    levels_sorted_idx.tolist(),
                    levels_sorted.tolist(),
                    diffs_sorted.tolist(),
                )
            )
            print(
                f"Iter: {ii:05}",
                f"\tLossE: {epoch_i_energy:.2f} ({epoch_i_energy_std:.2f})",
                f"\tK: {epoch_i_kinetic:.2f} ({epoch_i_kinetic_std:.2f})",
                f"\tV: {epoch_i_potential:.2f} ({epoch_i_potential_std:.2f})",
                f"\t\tac(pmove)={pmove_per_orb}\tstepsize={step_size}"
                f"\ttime= {(t2-t1):.2f}s"
                f"\tlevels_sorted(idx:abs(diff) cm^-1): {levels_stream_str}",
                flush=True,
            )

            f.write(
                f"{ii:06} {epoch_i_energy:.12f} {epoch_i_energy_std:.12f} "
                f"{epoch_i_kinetic:.12f} {epoch_i_kinetic_std:.12f} "
                f"{epoch_i_potential:.12f} {epoch_i_potential_std:.12f} "
            )
            for single_p in pmove_per_orb:
                f.write(f"{single_p:.2f} ")
            for single_step_size in step_size:
                f.write(f"{single_step_size:.2f} ")
            # Append sorted level differences (cm^-1) to the end of the line
            f.write(f"{t2 - t1:.9f} ")
            for v in diffs_sorted.tolist():
                f.write(f"{v:.1f} ")
            f.write("\n")
            f.flush()
            if training_args.mc_selfadjust_stepsize:
                # if ii % 1 == 0:
                step_size = adjust_mcmc_step_size(
                    step_size,
                    pmove_per_orb,
                    training_args.mc_adjust_upper,
                    training_args.mc_adjust_lower,
                    training_args.mc_adjust_increase_factor,
                    training_args.mc_adjust_decrease_factor,
                )

            if ii % training_args.ckpt_epochs == 0:
                if training_args.optimizer == "adam":
                    opt_state_ckpt = opt_state
                elif training_args.optimizer == "adamw":
                    opt_state_ckpt = opt_state
                ckpt = {
                    "training_args": training_args,
                    "key": key,
                    "x": x,
                    "opt_state": opt_state_ckpt,
                    "params_flow": params_flow,
                    "boltzmann_weights": boltzmann_factor,
                }
                save_ckpt_filename = ckpt_utils.ckpt_filename(ii, path)
                ckpt_utils.save_data(ckpt, save_ckpt_filename)
                print(f"save file: {save_ckpt_filename}", flush=True)
                print(
                    f"total time used: {(t2-t0):.2f}s ({((t2-t0)/3600):.2f}h),",
                    f"latest speed: {3600/(t2-t1):.1f} epochs per hour.",
                    flush=True,
                )


if __name__ == "__main__":
    training_kernel()

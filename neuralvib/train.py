"""Train"""

import os
import time
import subprocess
import argparse
import warnings
from datetime import datetime

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
from neuralvib.utils.key import key_batch_split
from neuralvib.utils.loss import Loss

from neuralvib.utils.mcmc import Metropolis, mcmc_pmap
from neuralvib.utils.mcmc import mcmc_pmap_ebes
from neuralvib.utils.update import Update
from neuralvib.utils.network_pretrain import FlowPretrain

# from neuralvib.wfbasis.basis import InvariantGaussian
from neuralvib.wfbasis.basis import HermiteFunction
from neuralvib.wfbasis.wf_ansatze import WFAnsatz

jax.config.update("jax_enable_x64", True)


def set_args():
    """Setting input args."""
    parser = argparse.ArgumentParser(
        description="simulation for the harmonic oscillator"
    )

    # folder to save data.
    parser.add_argument(
        "--folder",
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
        choices=["CH5+", "CH5+NoCarbon", "CH5+Jacobi"],
        help="molecule to compute, include toy models."
        "Note: the option User is for user specified potential files."
        "CH5+: the CH5+ molecule,"
        "CH5+-NoCarbon: the CH5+ molecule without carbon atom(hence carbon,"
        "is set to be origin of the coordinates)",
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

    parser.add_argument("--num_orb", type=int, default=1, help="number of orbitals")
    parser.add_argument(
        "--choose_orb", type=int, nargs="+", default=None, help="choose orbital"
    )

    # flow model: neural autoregressive flow & deep dense sigmoid flow parameters.
    parser.add_argument(
        "--flow_type",
        type=str,
        default="RNVP",
        choices=["MoleNet", "RNVP"],
        help="The flow type. NOTE: FermiNetCH4 is specified for CH4."
        "For other molecules, the equivariant flow should"
        "be desinated as MoleNet",
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
        "--ferminet_spsize",
        type=int,
        default=128,
        help="FermiNet single particle stream hidden units",
    )
    parser.add_argument(
        "--ferminet_tpsize",
        type=int,
        default=16,
        help="FermiNet two particle stream hidden units",
    )
    parser.add_argument(
        "--ferminet_init_stddev",
        type=float,
        default=0.0001,
        help="FermiNet init_stddev",
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
        "--clip_factor", type=float, default=5.0, help="clip factor for gradient"
    )

    parser.add_argument(
        "--ckpt_epochs", type=int, default=100, help="save checkpoint every ckpt_epochs"
    )
    parser.add_argument(
        "--excite_gen_type",
        type=int,
        required=True,
        help="Excitation generation type: 1 or 2."
        "corresponding to InitMolecule.generate_combinations"
        "or InitMolecule.generate_combinations_2",
    )

    # ========== args.params -> params ==========
    args = parser.parse_args()
    return args


def training_kernel() -> None:
    """Training Kernel"""
    # Get the current date and time
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d/%H:%M:%S")

    print("jax.__version__:", jax.__version__)
    print(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True
        ).stdout,
        flush=True,
    )

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

    print("\n========== Initialize Excitation Number ==========")
    # excitation_number = np.zeros(18, dtype=int)
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
        molecule=training_args.molecule,
        particles=molecule_init_obj.particles,
        particle_mass=molecule_init_obj.particle_mass,
    )
    print("x.shape:", x.shape)

    print("\n========== Initialize Wavefunction ==========")
    hermite_func_obj = HermiteFunction(
        molecule_init_obj,
    )
    wf_ansatze_obj = WFAnsatz(
        flow=flow,
        log_phi_base=hermite_func_obj.log_phi_base,
        training_args=training_args,
    )
    log_wf_ansatze = wf_ansatze_obj.log_wf_ansatz

    print("\n========== Initialize Metropolis ==========")
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
    scheduler = optax.exponential_decay(
        init_value=training_args.adam_lr, transition_steps=1000, decay_rate=0.99
    )
    # scheduler = optax.constant_schedule(training_args.adam_lr)
    if training_args.optimizer == "adam":
        optimizer = optax.adam(learning_rate=scheduler)
        print(f"Optimizer adam:\n    learning rate: {training_args.adam_lr}")
    elif training_args.optimizer == "adamw":
        optimizer = optax.adamw(
            learning_rate=training_args.adam_lr,
            weight_decay=training_args.adamw_weight_decay,
            nesterov=True,
        )
        print(f"Optimizer adam:\n    learning rate: {training_args.adam_lr}")
        print(f"\tweight_decay={training_args.adamw_weight_decay}")

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
            # key, x, probability_batched, step_size, pmove_per_orb = mcmc_pmap_ebes(
            #     training_args.mc_steps,
            #     subkey,
            #     x,
            #     excitation_number,
            #     params_flow,
            #     probability_batched,
            #     step_size,
            #     log_wf_ansatze,
            # )
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
                step_size = jnp.where(pmove_per_orb > 0.65, step_size * 1.05, step_size)
                step_size = jnp.where(pmove_per_orb < 0.25, step_size * 0.95, step_size)
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
        # if training_args.pretrain_network:
        #     raise NotImplementedError(
        #         "This part to use pretrain wf to initialize x"
        #         " is not implemented for multi-orbital calculation."
        #     )
        #     print("Using pretrain wf to initialize x, the former x is discarded.")
        #     hermite_smp_obj = flow_pretain_obj.hermite_func_sampler_obj
        #     hermite_smp_obj.batch = training_args.batch
        #     hermite_smp_obj.batch = training_args.batch // 120
        #     x = hermite_smp_obj.sampler()
        #     x = flow_pretain_obj.permute_atoms(x, molecule_init_obj.molecule)
        #     probability_batched = (
        #         jax.vmap(log_wf_ansatze, in_axes=(None, 0, None))(
        #             params_flow, x, excitation_numbers
        #         )
        #         * 2
        #     )
        #     del flow_pretain_obj
        #     del hermite_smp_obj
        #     # key = jax.random.split(key, num_devices)
        # else:
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
            # key, x, probability_batched, step_size, pmove_per_orb = mcmc_pmap_ebes(
            #     training_args.mc_steps,
            #     subkey,
            #     x,
            #     excitation_number,
            #     params_flow,
            #     probability_batched,
            #     step_size,
            #     log_wf_ansatze,
            # )
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
                step_size = jnp.where(pmove_per_orb > 0.65, step_size * 1.05, step_size)
                step_size = jnp.where(pmove_per_orb < 0.25, step_size * 0.95, step_size)
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
    )
    update_func = update_obj.update

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
            print(
                f"Iter: {ii:05}",
                f"\tLossE: {epoch_i_energy:.2f} ({epoch_i_energy_std:.2f})",
                f"\tK: {epoch_i_kinetic:.2f} ({epoch_i_kinetic_std:.2f})",
                f"\tV: {epoch_i_potential:.2f} ({epoch_i_potential_std:.2f})",
                f"\t\tac(pmove)={pmove_per_orb}\tstepsize={step_size}"
                f"\ttime= {(t2-t1):.2f}s",
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
            f.write(f"{t2 - t1:.9f} " "\n")
            f.flush()
            if training_args.mc_selfadjust_stepsize:
                # if ii % 1 == 0:
                step_size = jnp.where(pmove_per_orb > 0.65, step_size * 1.05, step_size)
                step_size = jnp.where(pmove_per_orb < 0.25, step_size * 0.95, step_size)

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

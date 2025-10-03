"""Update"""

from typing import Callable
from functools import partial
from argparse import Namespace

import numpy as np
import optax
import jax
import jax.numpy as jnp
from optax._src import base as optax_base

from neuralvib.utils.mcmc import mcmc_pmap_ebes
from neuralvib.utils.mcmc import mcmc_pmap


def clip_grad_norm(grads, max_norm):
    """Clips gradients based on their L2 norm."""
    global_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)])
    )
    clip_factor = max_norm / (global_norm + 1e-6)
    # JIT-compatible version using jnp.where instead of Python conditional
    clipped_grads = jax.tree.map(
        lambda x: x * jnp.where(global_norm > max_norm, clip_factor, 1.0), grads
    )
    return clipped_grads


def naive_fori_loop(lower, upper, body_fun, init_val):
    """A customly implemented fori_loop
    To be compatible with jax's api
    but avoiding JIT.
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


@partial(
    jax.jit,
    static_argnums=(0,),
)
def update_params(
    optimizer: optax.GradientTransformation,
    grad,
    opt_state: optax.OptState,
    params: optax_base.Params,
) -> tuple[optax_base.Params, optax.OptState]:
    """Updating parameters using optax

    Args:
        optimizer: the registered optimizer.
        grad: the gradient of the loss function.
        opt_state: the optimizer state (optax api).
        params: the parameters of the network.

    Returns:
        params: the updated parameters of the network.
        opt_state: the current opt_state that after update.
    """
    grad = clip_grad_norm(grad, max_norm=5.0)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


class Update:
    """Update functions"""

    def __init__(
        self,
        mcmc_steps: int,
        batch_size: int,
        num_orb: int,
        excitation_numbers: np.ndarray | jax.Array,
        loss_and_grad: Callable,
        optimizer: optax.GradientTransformation,
        acc_steps: int,
        fori_loop_init: dict,
        log_wf_ansatz: Callable,
        batch_info: dict,
        metropolis_sampler_batched: Callable,
        training_args: Namespace,
    ) -> None:
        self.mcmc_steps = mcmc_steps
        self.batch_size = batch_size
        self.num_orb = num_orb
        self.excitation_numbers = excitation_numbers
        self.loss_and_grad = loss_and_grad
        self.optimizer = optimizer
        self.acc_steps = acc_steps
        self.loss_init = fori_loop_init["loss_init"]
        self.energies_init = fori_loop_init["energies_init"]
        self.grad_init = fori_loop_init["grad_init"]
        self.log_wf_ansatz = log_wf_ansatz
        self.metropolis_sampler_batched = metropolis_sampler_batched
        self.num_devices = batch_info["num_devices"]
        self.batch_per_device = batch_info["batch_per_device"]
        self.training_args = training_args

    def update(
        self,
        key: jax.Array,
        xs_batched: jax.Array,
        probability_batched: jax.Array,
        mc_step_size: jax.Array,
        params: jax.Array | np.ndarray | dict | optax_base.Params,
        opt_state: optax.OptState,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        jax.Array,
        jax.Array,
        optax_base.Params,
        optax.OptState,
        jax.Array,
        jax.Array,
    ]:
        """Single update
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
            opt_state: the optimizer state.

        Returns:
            loss: the meaned energy as loss
            energy_std: the energy standard deviation.
            kinetic: the meaned kinetic energy.
            kinetic_std: the kinetic standard deviation.
            potential: the meaned potential energy.
            potential_std: the potential standard deviation.
            xs_batched: the coordinate
            probability_batched: the batched probability (wavefunction**2)
            params: the flow parameters
            opt_state: the optimizer state.
            pmove_per_orb: (num_orb,) the portion of moved
                particles in last mcmc step.
                NOTE: this is a per orbital property!
            mc_step_size: (num_orb,) updated mcmc moving step size.
                NOTE: this is a per orbital property!

        """
        # print("⚠️Recompile Indicator⚠️")

        def _acc_body_func(i, val):
            """Gradient Accumulation Body Function"""
            (
                loss,
                energies,
                kinetics,
                potentials,
                grad,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_in,
                params,
            ) = val
            # (
            #     key,
            #     xs_batched,
            #     probability_batched,
            #     mc_step_size,
            #     pmove_per_orb,
            # ) = mcmc_pmap_ebes(
            #     self.mcmc_steps,
            #     key,
            #     xs_batched,
            #     self.excitation_number,
            #     params,
            #     probability_batched,
            #     mc_step_size,
            #     self.log_wf_ansatz,
            # )
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
            loss_i, energies_i, kinetics_i, potentials_i, gradients_i = (
                self.loss_and_grad(params, xs_batched)
            )
            grad_i = gradients_i
            (loss, energies, kinetics, potentials, grad) = jax.tree.map(
                lambda acc, i: acc + i,
                (loss, energies, kinetics, potentials, grad),
                (loss_i, energies_i, kinetics_i, potentials_i, grad_i),
            )
            return (
                loss,
                energies,
                kinetics,
                potentials,
                grad,
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
            jnp.float64(0.0),
            jnp.zeros_like(energies_init),
            jnp.zeros_like(energies_init),
            jnp.zeros_like(energies_init),
            jax.tree.map(jnp.zeros_like, self.grad_init),
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            jnp.zeros(self.num_orb),
            params,
        )

        (
            loss,
            energies,
            kinetics,
            potentials,
            grad,
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            pmove_per_orb,
            params,
        ) = naive_fori_loop(0, self.acc_steps, _acc_body_func, acc_init_val)

        (loss, energies, kinetics, potentials, grad) = jax.tree.map(
            lambda acc: acc / self.acc_steps,
            (loss, energies, kinetics, potentials, grad),
        )
        # energies, kinetics, potentials: (num_devices,batch_per_device, num_orb)
        energies = jnp.sum(energies, axis=-1)
        kinetics = jnp.sum(kinetics, axis=-1)
        potentials = jnp.sum(potentials, axis=-1)
        kinetic = jnp.mean(kinetics)
        potential = jnp.mean(potentials)

        params, opt_state = update_params(self.optimizer, grad, opt_state, params)
        energy_std = jnp.sqrt(
            ((energies**2).mean() - loss**2) / (self.acc_steps * self.batch_size)
        )
        kinetic_std = jnp.sqrt(
            ((kinetics**2).mean() - kinetic**2) / (self.acc_steps * self.batch_size)
        )
        potential_std = jnp.sqrt(
            ((potentials**2).mean() - potential**2) / (self.acc_steps * self.batch_size)
        )
        xs_batched = xs_batched.reshape(
            self.training_args.batch,
            self.training_args.num_orb,
            self.training_args.num_of_particles,
            self.training_args.dim,
        )
        probability_batched = probability_batched.reshape(
            self.training_args.batch, self.training_args.num_orb
        )
        return (
            loss,
            energy_std,
            kinetic,
            kinetic_std,
            potential,
            potential_std,
            xs_batched,
            probability_batched,
            params,
            opt_state,
            pmove_per_orb,
            mc_step_size,
        )

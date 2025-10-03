"""Loss"""

from threading import local
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class Loss:
    """The original total loss function
    as 'it is' and the custom loss function, custom_loss
    ONLY for Gradient Estimator!

    Attributes:
        self.batch_local_energy: the batched local energy estimator.
        self.batch_wf: the logg domain wavefunction callable
            vmapped one BATCH!
        self.clip_factor: the gradient clipping factor for energy part.
    """

    def __init__(
        self,
        wf_ansatz: Callable,
        batched_local_energy_estimator: Callable,
        excitation_numbers: np.ndarray | jax.Array,
        clip_factor: float,
    ) -> None:
        """Init

        Args:
            wf_ansatz: function, signature (params, xs,), which evaluates wavefunction
                at a single MCMC configuration given the network parameters.
            batched_local_energy_estimator: signature
                (params, batched_xs, excitation_numbers),here
                xs refers to the x coordinates in one single batch, xs has shape
                (num_orb,num_of_particles,dim,)
                and excitation_numbers has shape (num_orb,num_particles*dim,)
                and each return item has shape (batch,num_orb)
            excitation_numbers: (num_orb,num_particles*dim,)
                the corresponding excitation
                quantum number of all excitation states of
                each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
            clip_factor: the gradient clipping factor for energy part.
        """
        self.batch_local_energy = batched_local_energy_estimator
        self.excitation_numbers = excitation_numbers
        log_wf = wf_ansatz
        # batch_wf: signature (params, batched_xs, excitation_numbers)
        self.batch_wf = jax.vmap(
            jax.vmap(log_wf, in_axes=(None, 0, 0)), in_axes=(None, 0, None), out_axes=0
        )
        self.clip_factor = clip_factor

    def total_energy(
        self, params: dict, batched_xs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Total energy of an ensemble (batched)

        Args:
            params: the network parameters
            batched_xs: (num_of_batch,num_orb, num_of_particles,dim,)
                the batched configuration cartesian coordinate of the particle(s).

        Returns:
            loss: the total loss of the system, meaned over batch.
            e_l: (batch,num_orb) the local energy of each state.
            kinetics: (batch,num_orb) the local kinetics of each state.
            potentials: (batch,num_orb) the potentials of each state.
        """
        kinetics, potentials = self.batch_local_energy(
            params, batched_xs, self.excitation_numbers
        )
        e_l = kinetics + potentials
        loss = jnp.mean(jnp.sum(e_l, axis=-1))
        return loss, e_l, kinetics, potentials

    # Pmap should be wrapped when use
    # but not here as wrapper.
    def loss_and_grad_pmap(
        self,
        params: dict,
        batched_xs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """Manually implemented loss_and_grad to
        compatible with previous FermiNet style loss,
        jax.value_and_grad(loss_energy, argnums=0, has_aux=True)

        Should be used as
        ```python
        jax.pmap(
            loss_obj.loss_and_grad_pmap,
            axis_name="xla_device",
            in_axes=(None, 0),
            out_axes=(None, 0, 0, 0, None),
        )
        ```

        NOTE: The shapes here are SUPPOSED that the
            function is pmapped!

        Args:
            params: the network parameters
            batched_xs: (num_device, batch_per_device, num_orb, num_of_particles,dim,)
                the batched configuration cartesian coordinate of the particle(s).

        Returns:
            loss: (1,) the total loss
            energies: (num_device,batch_per_device, num_orb) the local energies
            kinetics: (num_device,batch_per_device, num_orb) the local kinetics
                of each state.
            potentials: (num_device,batch_per_device, num_orb) the potentials
                of each state.
            gradients: dict, the gradients to network parameters.
        """

        def _batch_wf(params, batched_xs):
            """Return shape (batch, num_orb)"""
            return self.batch_wf(params, batched_xs, self.excitation_numbers)

        loss, energies, kinetics, potentials = self.total_energy(params, batched_xs)
        local_energies = jax.lax.stop_gradient(energies)

        def _custom_loss(params, batched_xs):
            # local_energis: (batch,num_orb)
            # logpsix: (batch, num_orb)
            logpsix = _batch_wf(params, batched_xs)
            energies_batch_average = jnp.mean(local_energies, axis=0)  # (num_orb,)
            energies_device_average = jax.lax.pmean(
                energies_batch_average, axis_name="xla_device"
            )  # (num_orb,)

            # For Control Variate and clipping
            # Clipping may be important for nodal area!
            clip_factor = self.clip_factor
            tv = jnp.mean(
                jnp.abs(local_energies - energies_device_average), axis=0
            )  # (num_orb,)
            tv = jax.lax.pmean(tv, axis_name="xla_device")  # (num_orb,)

            # jax.debug.print(
            #     "local_energies: {local_energies}, ",local_energies=local_energies)
            # jax.debug.print("local_energies.shape {shape}",shape=local_energies.shape)
            # jax.debug.print(
            #     "tv: {tv}, ",   tv=tv)
            assert tv.shape[0] == local_energies.shape[1]

            local_energies_clipped = jnp.clip(
                local_energies,
                energies_device_average - clip_factor * tv,
                energies_device_average + clip_factor * tv,
            )  # (batch, num_orb)
            custom_loss = 2 * jnp.sum(
                jax.lax.pmean(
                    jnp.mean(
                        (logpsix * (local_energies_clipped - energies_device_average)),
                        axis=0,
                    ),
                    axis_name="xla_device",
                )
            )
            return custom_loss

        gradients_func = jax.grad(_custom_loss, argnums=0)
        gradients = gradients_func(params, batched_xs)
        loss = jax.lax.pmean(loss, axis_name="xla_device")
        return (loss, energies, kinetics, potentials, gradients)

    def loss_and_grad(
        self,
        params: dict,
        batched_xs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], dict]:
        """Manually implemented loss_and_grad to
        compatible with previous FermiNet style loss,
        jax.value_and_grad(loss_energy, argnums=0, has_aux=True)

        Args:
            params: the network parameters
            batched_xs: (num_of_batch, num_of_particles,dim,)
                the batched configuration cartesian coordinate of the particle(s).

        Returns:
            loss: the total loss
            energies: (batch, ) the local energies
            kinetics: (batch,) the local kinetics of each state.
            potentials: (batch,) the potentials of each state.
            gradients: dict, the gradients to network parameters.
        """
        raise DeprecationWarning(
            "This function is deprecated, please use loss_and_grad_pmap instead."
        )

        def _batch_wf(params, batched_xs):
            return self.batch_wf(params, batched_xs, self.excitation_number)

        loss, energies, kinetics, potentials = self.total_energy(params, batched_xs)
        local_energies = jax.lax.stop_gradient(energies)

        def _custom_loss(params, batched_xs):
            # local_energis: (batch,)
            # logpsix: (batch, )
            logpsix = _batch_wf(params, batched_xs)
            energies_batch_average = jnp.mean(local_energies, axis=0)  # (1,)

            # For Control Variate and clipping
            # Clipping may be important for nodal area!
            clip_factor = self.clip_factor
            tv = jnp.mean(
                jnp.abs(local_energies - energies_batch_average), axis=0
            )  # (1,)
            local_energies_clipped = jnp.clip(
                local_energies,
                energies_batch_average - clip_factor * tv,
                energies_batch_average + clip_factor * tv,
            )
            custom_loss = 2 * jnp.sum(
                jnp.mean(
                    (logpsix * (local_energies_clipped - energies_batch_average)),
                    axis=0,
                )
            )
            return custom_loss

        gradients_func = jax.grad(_custom_loss, argnums=0)
        gradients = gradients_func(params, batched_xs)
        return ((loss, energies, kinetics, potentials), gradients)

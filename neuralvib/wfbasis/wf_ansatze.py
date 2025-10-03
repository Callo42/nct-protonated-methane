"""Wavefunction ansatze"""

import argparse
from typing import Callable
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from optax._src import base as optax_base


class WFAnsatz:
    """The flow wave function ansatz

    Attributes:
        self.flow: the normalizing flow network
        self.log_phi_base: callable that recieves system's coordinates
            and returns the wave function base in log domain
        self.vscf_coeff: the vscf coefficients if provided
    """

    def __init__(
        self,
        flow: hk.Transformed,
        log_phi_base: Callable,
        training_args: argparse.Namespace,
        vscf_coeff: np.ndarray | None = None,
    ) -> None:
        """Init wavefunction ansatze object

        Args:
            flow: the haiku transformed flow network
            log_phi_base: callable that recieves system's coordinates
                and returns the wave function base in log domain
            vscf_coeff: the vscf coefficients if provided.
            training_args: the input arguments when invoking the program.
        """
        self.flow = flow
        self.log_phi_base = log_phi_base
        self.vscf_coeff = vscf_coeff
        if training_args.flow_type == "RNVP":
            self.log_wf_ansatz = self.log_wf_ansatz_with_logjac
        else:
            self.log_wf_ansatz = self.log_wf_ansatz_direct_logjac

    def log_wf_ansatz_direct_logjac(
        self,
        params: jax.Array | np.ndarray | optax_base.Params,
        x: jax.Array | np.ndarray,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """The flow transformed log wavefunction
        using direct log jacobian determinant
        which refers to calling jax.jacfwd to compute the jacobian.
        And the flow only returns x, no additional jacobian determinant.

        Args:
            params: the flow parameter
            x:(num_of_particles,dim) the coordinate before flow
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            log_amplitude: float, the log wavefunction
                log|psi|
        """
        z = self.flow.apply(params, None, x)
        log_phi = self.log_phi_base(z, excitation_number)

        n, dim = x.shape
        x_flatten = x.reshape(-1)

        def _flow_flatten(x_flatten: jax.Array):
            return self.flow.apply(params, None, x_flatten.reshape(n, dim)).reshape(-1)

        jac = jax.jacfwd(_flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)

        log_amplitude = log_phi.real + 0.5 * logjacdet
        return log_amplitude

    def log_wf_ansatz_with_logjac(
        self,
        params: jax.Array | np.ndarray | optax_base.Params,
        x: jax.Array | np.ndarray,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """The flow transformed log wavefunction
        with flow returning both x and log jacobian determinant.

        Args:
            params: the flow parameter
            x:(num_of_particles,dim) the coordinate before flow
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).

        Returns:
            log_amplitude: float, the log wavefunction
                log|psi|
        """
        z, logjacdet = self.flow.apply(params, None, x)
        log_phi = self.log_phi_base(z, excitation_number)

        log_amplitude = log_phi.real + 0.5 * logjacdet
        return log_amplitude

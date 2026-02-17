"""Wavefunction ansatze"""

import argparse
from typing import Callable
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from optax._src import base as optax_base

from neuralvib.utils.frame import fix_eckart_frame
from neuralvib.utils.mcmc import calculate_center_mass_coor


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
        ms: np.ndarray,
        x_ref: np.ndarray,  # equilibrium geometry, possibly flattened
        vscf_coeff: np.ndarray | None = None,
    ) -> None:
        """Init wavefunction ansatz object

        Args:
            flow: the haiku transformed flow network
            log_phi_base: callable that recieves system's coordinates
                and returns the wave function base in log domain
            ms: masses; can be (N,), (N,1), (N,3) or (3N,)
            x_ref: equilibrium geometry; can be (N,3) or flattened (3N,)
            vscf_coeff: the vscf coefficients if provided.
            training_args: the input arguments when invoking the program.
        """
        self.flow = flow
        self.log_phi_base = log_phi_base
        self.vscf_coeff = vscf_coeff

        # ---- normalize x_ref to shape (N, 3) ----
        x_ref_j = jnp.asarray(x_ref)

        if x_ref_j.ndim == 1:
            # flattened: length must be 3N
            if x_ref_j.size % 3 != 0:
                raise ValueError(
                    f"x_ref is 1D with size {x_ref_j.size}, "
                    "which is not divisible by 3; expected 3*N."
                )
            N = x_ref_j.size // 3
            dim = 3
            x_ref_reshaped = x_ref_j.reshape((N, dim))
        elif x_ref_j.ndim == 2:
            # already (N, 3) ideally
            if x_ref_j.shape[1] != 3:
                raise ValueError(
                    f"x_ref has shape {x_ref_j.shape}, expected (N, 3) "
                    "or flattened length 3N."
                )
            x_ref_reshaped = x_ref_j
            N, dim = x_ref_reshaped.shape
        else:
            raise ValueError(
                f"x_ref must be 1D or 2D, got shape {x_ref_j.shape} (ndim={x_ref_j.ndim})."
            )

        self.n_particles = N
        self.dim = dim  # should be 3
        # store unflattened ref
        x_ref_j = x_ref_reshaped  # (N,3)

        # ---- normalize masses to shape (N, 3) for CoM function ----
        ms_j = jnp.asarray(ms)

        if ms_j.ndim == 1:
            # Case 1: one mass per atom: (N,)
            if ms_j.size == N:
                ms_full = jnp.repeat(ms_j[:, None], dim, axis=1)   # (N,3)
            # Case 2: per-coordinate / flattened: (3N,)
            elif ms_j.size == N * dim:
                ms_full = ms_j.reshape((N, dim))                   # (N,3)
            else:
                raise ValueError(
                    f"Incompatible ms shape {ms_j.shape} for x_ref with N={N}, dim={dim}."
                )
        elif ms_j.ndim == 2:
            if ms_j.shape == (N, dim):
                ms_full = ms_j
            elif ms_j.shape == (N, 1):
                ms_full = jnp.repeat(ms_j, dim, axis=1)           # (N,3)
            else:
                raise ValueError(
                    f"Incompatible ms shape {ms_j.shape} for x_ref with shape {x_ref_j.shape}."
                )
        else:
            raise ValueError(
                f"ms must be 1D or 2D, got shape {ms_j.shape} (ndim={ms_j.ndim})."
            )

        self.ms = ms_full  # (N,3)

        # ---- Precompute COM-centered reference geometry for Eckart alignment ----
        # This uses your existing CoM function unchanged.
        com_ref = calculate_center_mass_coor(self.ms, x_ref_j)     # (3,)
        self.x_ref_centered = x_ref_j - com_ref                    # (N,3)

        # ---- Flow type dispatch ----
        if training_args.flow_type == "RNVP":
            self.log_wf_ansatz = self.log_wf_ansatz_with_logjac
        else:
            raise NotImplementedError(
                f"flow type {training_args.flow_type} not implemented yet."
            )


    def log_wf_ansatz_with_logjac(
        self,
        params: jax.Array | np.ndarray | optax_base.Params,
        x: jax.Array | np.ndarray,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """
        Flow-transformed log wavefunction in an Eckart-aligned, COM-free frame.

        - Translation invariance via COM removal.
        - Rotation (SO(3)) invariance via mass-weighted Kabsch alignment
        to a fixed reference geometry (self.x_ref_centered).
        """
        x = jnp.asarray(x)

        # ---- reshape x to (N, 3) if flattened ----
        if x.ndim == 1:
            N = self.n_particles
            dim = self.dim
            if x.size != N * dim:
                raise ValueError(
                    f"x is 1D with size {x.size}, expected {N*dim} (N*dim) "
                    f"for N={N}, dim={dim}."
                )
            x = x.reshape((N, dim))  # (N,3)
        elif x.ndim == 2:
            if x.shape != (self.n_particles, self.dim):
                raise ValueError(
                    f"x has shape {x.shape}, expected {(self.n_particles, self.dim)}."
                )
        else:
            raise ValueError(
                f"x must be 1D or 2D, got shape {x.shape} (ndim={x.ndim})."
            )

        # Apply the shared "frame fix" used across the codebase:
        # COM removal + mass-weighted Kabsch alignment to `self.x_ref_centered`.
        x_eckart = fix_eckart_frame(
            x,
            ms=self.ms,
            x_ref_centered=self.x_ref_centered,
        )  # (N, 3)

        # 5) Apply flow in the Eckart frame
        z, logjacdet = self.flow.apply(params, None, x_eckart)
        log_phi = self.log_phi_base(z, excitation_number)

        # R is orthogonal => |det R| = 1, no extra log|det| term
        log_amplitude = log_phi.real + 0.5 * logjacdet
        return log_amplitude

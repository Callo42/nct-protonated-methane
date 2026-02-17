"""Coordinate frame utilities.

Centralizes the "frame fix" logic used across initialization, MCMC, and the
wavefunction ansatz.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.utils.mcmc import calculate_center_mass_coor


def _as_n3(x: jax.Array | np.ndarray) -> jax.Array:
    """Normalize an input configuration to shape (N, 3)."""
    x = jnp.asarray(x)
    if x.ndim == 1:
        if x.size % 3 != 0:
            raise ValueError(
                f"Expected flattened coordinates with length 3*N, got {x.size}."
            )
        x = x.reshape((x.size // 3, 3))
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3); got {x.shape}.")
    return x


def _normalize_masses(ms: jax.Array | np.ndarray, n_particles: int) -> jax.Array:
    """Normalize masses to shape (N, 3) for COM and mass-weighting."""
    ms = jnp.asarray(ms)
    if ms.ndim == 1:
        if ms.size == n_particles:
            return jnp.repeat(ms[:, None], 3, axis=1)
        if ms.size == n_particles * 3:
            return ms.reshape((n_particles, 3))
        raise ValueError(
            f"Incompatible mass vector length {ms.size} for N={n_particles}."
        )
    if ms.ndim == 2:
        if ms.shape == (n_particles, 3):
            return ms
        if ms.shape == (n_particles, 1):
            return jnp.repeat(ms, 3, axis=1)
        raise ValueError(f"Incompatible mass shape {ms.shape} for N={n_particles}.")
    raise ValueError(f"Masses must be 1D or 2D, got shape {ms.shape}.")


def fix_eckart_frame(
    xs: jax.Array | np.ndarray,
    ms: jax.Array | np.ndarray,
    *,
    x_ref: jax.Array | np.ndarray | None = None,
    x_ref_centered: jax.Array | np.ndarray | None = None,
    enforce_proper_rotation: bool = False,
) -> jax.Array:
    """Fix translation and rotation using a COM-free, Eckart-aligned frame.

    This matches the workflow currently used in `WFAnsatz.log_wf_ansatz_with_logjac`:
    - subtract center of mass (translation invariance)
    - mass-weighted Kabsch alignment to a fixed reference geometry (rotation invariance)
    - rotate into the body-fixed (Eckart) frame via `xs_centered @ R`

    Args:
        xs: coordinates with shape (..., N, 3) or flattened (3N,).
        ms: masses with shape (N,), (N,1), (N,3), or flattened (3N,).
        x_ref: reference geometry with shape (N, 3) or flattened (3N,).
            Required if `x_ref_centered` is not provided.
        x_ref_centered: COM-centered reference geometry with shape (N, 3).
        enforce_proper_rotation: if True, enforce det(R)=+1 (SO(3)).

    Returns:
        xs_eckart: coordinates in a COM-free, Eckart-aligned frame with shape
            (..., N, 3).
    """
    xs = jnp.asarray(xs)
    if xs.ndim == 1:
        xs = _as_n3(xs)
    if xs.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {xs.shape}.")

    n_particles = int(xs.shape[-2])
    ms_full = _normalize_masses(ms, n_particles)  # (N,3)
    ms_1d = ms_full[:, 0]  # (N,)

    if x_ref_centered is None:
        if x_ref is None:
            raise ValueError("Provide either `x_ref_centered` or `x_ref`.")
        x_ref_n3 = _as_n3(x_ref)
        if x_ref_n3.shape[0] != n_particles:
            raise ValueError(
                f"`x_ref` has N={x_ref_n3.shape[0]}, expected N={n_particles}."
            )
        com_ref = calculate_center_mass_coor(ms_full, x_ref_n3)  # (3,)
        x_ref_centered_j = x_ref_n3 - com_ref  # (N,3)
    else:
        x_ref_centered_j = _as_n3(x_ref_centered)
        if x_ref_centered_j.shape[0] != n_particles:
            raise ValueError(
                f"`x_ref_centered` has N={x_ref_centered_j.shape[0]}, expected N={n_particles}."
            )

    com = calculate_center_mass_coor(ms_full, xs)  # (...,3)
    xs_centered = xs - com[..., None, :]  # (...,N,3)

    sqrt_m = jnp.sqrt(ms_1d)[:, None]  # (N,1)
    Xw = xs_centered * sqrt_m  # (...,N,3)
    Qw = x_ref_centered_j * sqrt_m  # (N,3)

    # Kabsch alignment: solve argmin_R || Xw R - Qw || with R in O(3) (or SO(3)).
    C = jnp.swapaxes(Xw, -2, -1) @ Qw  # (...,3,3)
    U, _, Vh = jnp.linalg.svd(C, full_matrices=False)
    R = U @ Vh  # (...,3,3)

    if enforce_proper_rotation:
        detR = jnp.linalg.det(R)  # (...,)
        sgn = jnp.where(detR < 0.0, -1.0, 1.0)
        corr = jnp.stack([jnp.ones_like(sgn), jnp.ones_like(sgn), sgn], axis=-1)
        R = (U * corr[..., None, :]) @ Vh

    xs_eckart = xs_centered @ R  # (...,N,3)
    return xs_eckart


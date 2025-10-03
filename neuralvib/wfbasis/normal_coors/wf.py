"""Wavefunction in normal coordinates"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def hermite(n: int, x: float) -> jax.Array:
    """The hermite polynominals"""
    h0 = 1.0 / jnp.pi ** (1 / 4)
    h1 = jnp.sqrt(2.0) * x / jnp.pi ** (1 / 4)

    def body_fun(i, val):
        valm2, valm1 = val
        return valm1, jnp.sqrt(2.0 / i) * x * valm1 - jnp.sqrt((i - 1) / i) * valm2

    _, hn = jax.lax.fori_loop(2, n + 1, body_fun, (h0, h1))

    return jax.lax.cond(n > 0, lambda: hn, lambda: h0)


@hermite.defjvp
def hermite_jvp(n: int, primals, tangents):
    (x,) = primals
    (dx,) = tangents
    hn = hermite(n, x)
    dhn = jnp.sqrt(2 * n) * hermite((n - 1) * (n > 0), x) * dx
    primals_out, tangents_out = hn, dhn
    return primals_out, tangents_out


def log_wf_base_1d(x: float, w: float, n: int) -> jax.Array:
    """The wave function ansatz (Gaussian)
    NOTE: 1D! The eigenstates of a one-dimensional
    harmonic oscillator with frequency w.

    Args:
        x: the 1D coordinate of the (single) mode.
        n: the excitation quantum number

        NOTE: n=0 for GS!

    Returns:
        log_psi: the log probability amplitude at x.
    """
    # for normal coordinates, the so-called mass
    # would always be 1.0
    # remain m here for future convience.
    m = 1.0

    log_psi = (
        jnp.log(m * w) / 4
        - 0.5 * m * w * x**2
        + jnp.log(jnp.abs(hermite(n, jnp.sqrt(m * w) * x)))
    )
    return log_psi


def log_wf_basis(
    xs: jax.Array,
    ws: jax.Array,
    indices: np.ndarray,
) -> jax.Array:
    """The log wavefunction basis
    Phi_{n1,n2}(z1,z2) = phi_n1(z1) * phi_n2(z2)
    NOTE: here ni refers to the quantum state number
        of i-th normal coordinate.
    NOTE: So this is a single eigenstate of a system with num_of_modes
        normal coordinates. And the state is denoted as
        N = {n1,n2} where n1 is the quantum number of the first mode
            and n2 the second mode.
        For example, for a system with 3 normal modes, the
            ground state (GS) would have
            xs = [x1,x2,x3]
            ws = [w1,w2,w3]
            indices = [0,0,0]
        And the wave function of the GS of the system would be
        Phi_0(x1,x2,x3) = Phi_{0,0,0}(x1,x2,x3)
                        = phi_0(x1) phi_0(x2) phi_0(x3)

    Args:
        xs: (num_of_modes,1) the normal coordinates
        ws: (num_of_modes,) the harmonic frequencies of corresponding
            modes
        indices: (num_of_modes,) the excitation quantum number
            of this single state
            of each mode

    Returns:
        log_amplitude: the log wavefunction
            log|Psi|
    """
    xs = xs.flatten()
    log_phis = jax.vmap(log_wf_base_1d)(xs, ws, indices)
    log_amplitude = jnp.sum(log_phis)
    return log_amplitude

"""MCMC"""

from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from optax._src import base as optax_base

from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.key import key_batch_split


def _recip_ratio(log_ratio):
    """Computing recip ratio
    From
    https://code.itp.ac.cn/wanglei/hydrogen/-/blob/500d1f6a4508597747a6c946331c45b130d0d2d4/src/mcmc.py
    """
    return jnp.exp(-jnp.minimum(0, log_ratio))


def _compute_ar(log_ratio, key_accept):
    """Computing accept and recip
    From
    https://code.itp.ac.cn/wanglei/hydrogen/-/blob/500d1f6a4508597747a6c946331c45b130d0d2d4/src/mcmc.py
    """
    ratio = jnp.exp(log_ratio)
    accept = jax.random.uniform(key_accept, ratio.shape) < ratio
    recip = _recip_ratio(log_ratio)

    return accept, recip


def calculate_center_mass_coor(
    ms: jax.Array | np.ndarray, xs: jax.Array | np.ndarray
) -> jax.Array:
    """Calculate the center of mass coordinate of a batch of particles.

    Args:
        ms: (num_of_particles, 3) the mass of each particle.
        xs: (num_of_particles, 3) the cartesian coordinates of each particle.

    Returns:
        com: (3,) the center of mass coordinate.
    """
    ms = jnp.array(ms)
    xs = jnp.array(xs)
    com = jnp.sum(xs * ms, axis=0) / jnp.sum(ms, axis=0)
    return com


def _align_to_axes(xs: jax.Array) -> jax.Array:
    """Rotate the coordinates to a standard orientation.
    1. The first particle is on the z-axis.
    2. The second particle is on the x-z plane.
    Handles edge cases where particles are at the origin or aligned with axes.
    """
    epsilon = 1e-8

    # --- Step 1: Rotate first particle to z-axis ---
    r1 = xs[0]
    r1_norm = jnp.linalg.norm(r1)

    z_axis = jnp.array([0.0, 0.0, 1.0])
    # Avoid division by zero if r1 is at origin
    r1_u = r1 / jnp.where(r1_norm < epsilon, 1.0, r1_norm)

    v = jnp.cross(r1_u, z_axis)
    c = jnp.dot(r1_u, z_axis)
    s = jnp.linalg.norm(v)

    # Rodrigues' rotation formula components
    # To avoid nan from division by zero if s=0
    s_safe = jnp.where(s < epsilon, 1.0, s)
    v_x = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R1_regular = jnp.eye(3) + v_x + v_x @ v_x * (1 - c) / (s_safe**2)

    # Handle case where r1 is already aligned or anti-aligned with z-axis
    # if c > 0, aligned, R1=I.
    # if c < 0, anti-aligned, R1 is 180 deg rot around x-axis.
    R1_aligned = jnp.diag(jnp.array([1.0, -1.0, -1.0]))
    R1_aligned = jnp.where(c > 0, jnp.eye(3), R1_aligned)

    R1 = jnp.where(s > epsilon, R1_regular, R1_aligned)

    # if r1 is near origin, no rotation at all.
    R1 = jnp.where(r1_norm > epsilon, R1, jnp.eye(3))

    xs_rotated1 = (R1 @ xs.T).T

    if xs.shape[0] < 2:
        return xs_rotated1

    # --- Step 2: Rotate second particle to x-z plane ---
    r2 = xs_rotated1[1]
    norm_xy = jnp.linalg.norm(r2[:2])

    norm_xy_safe = jnp.where(norm_xy < epsilon, 1.0, norm_xy)
    cos_phi = r2[0] / norm_xy_safe
    sin_phi = r2[1] / norm_xy_safe

    # Rotation around z-axis to bring y to 0.
    # This is rotation by -phi.
    # Rz(-phi) = [[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0,0,1]]
    R2_regular = jnp.array(
        [[cos_phi, sin_phi, 0.0], [-sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
    )

    R2 = jnp.where(norm_xy > epsilon, R2_regular, jnp.eye(3))

    xs_rotated2 = (R2 @ xs_rotated1.T).T

    return xs_rotated2


def rotate_to_eckart_frame(
    xs: jax.Array, reference_config: jax.Array, masses: jax.Array
) -> jax.Array:
    """Rotate configuration to the Eckart frame.

    Args:
        xs: (num_of_particles, 3) Cartesian coordinates
        reference_config: (num_of_particles, 3) Reference configuration
        masses: (num_of_particles, 3) Masses for each particle and dimension

    Returns:
        xs_rotated: (num_of_particles, 3) Rotated configuration
    """
    # Ensure both configurations are centered at center of mass
    xs_centered = xs - calculate_center_mass_coor(masses, xs)[None, :]
    ref_centered = (
        reference_config - calculate_center_mass_coor(masses, reference_config)[None, :]
    )

    # Compute the mass-weighted covariance matrix
    masses_sqrt = jnp.sqrt(masses)
    xs_weighted = xs_centered * masses_sqrt
    ref_weighted = ref_centered * masses_sqrt

    # Compute covariance matrix
    covariance = xs_weighted.T @ ref_weighted

    # Singular value decomposition
    u, _, vt = jnp.linalg.svd(covariance, full_matrices=False)

    # Ensure proper rotation (determinant = 1)
    det = jnp.linalg.det(u @ vt)
    correction = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, det]])

    # Compute rotation matrix
    rotation = u @ correction @ vt

    # Apply rotation
    xs_rotated = (rotation @ xs_centered.T).T

    return xs_rotated


def mcmc_ebes_kernal(
    logp_fn: Callable,
    x_init: jax.Array | np.ndarray,
    key: jax.Array,
    mc_steps: int,
    mc_width: float,
    logp_init: jax.Array | np.ndarray | None = None,
):
    """
        Markov Chain Monte Carlo (MCMC) sampling algorithm
        with electron-by-electron sampling (EBES).
        From
        https://code.itp.ac.cn/wanglei/hydrogen/-/blob/500d1f6a4508597747a6c946331c45b130d0d2d4/src/mcmc.py

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (..., n, dim).
        x_init: initial value of x, with shape (..., n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal, also denoted as mc_stddev in
            previous version.
        logp_init: initial logp (...,)

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """

    print("-----MC Algorithm: EBES-------")

    def single_step(ii, state):
        x, logp, key, accept_rate, recip_ratio = state
        key, key_proposal, key_accept = jax.random.split(key, 3)

        batchshape = x.shape[:-2]
        dim = x.shape[-1]
        x_move = jax.random.normal(key_proposal, (*batchshape, dim))
        x_proposal = x.at[..., ii, :].add(mc_width * x_move)
        logp_proposal = logp_fn(x_proposal)  # batchshape

        log_ratio = logp_proposal - logp
        accept, recip = _compute_ar(log_ratio, key_accept)
        accept_rate += accept.mean()
        recip_ratio += recip.mean()

        x_new = jnp.where(accept[..., None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)

        return x_new, logp_new, key, accept_rate, recip_ratio

    def step(i, state):
        x, logp, key, accept_rate, recip_ratio = state

        n = x.shape[-2]
        x_new, logp_new, key, accept_rate, recip_ratio = jax.lax.fori_loop(
            0, n, single_step, (x, logp, key, accept_rate, recip_ratio)
        )
        return x_new, logp_new, key, accept_rate, recip_ratio

    if logp_init is None:
        logp_init = logp_fn(x_init)

    x, logp, key, accept_rate, recip_ratio = jax.lax.fori_loop(
        0, mc_steps, step, (x_init, logp_init, key, 0.0, 0.0)
    )
    n = x.shape[-2]
    accept_rate /= mc_steps * n
    recip_ratio /= mc_steps * n
    return x, logp, key, accept_rate, recip_ratio


class Metropolis:
    """The Metropolis Algorithm
    iterating the positions of MCMC walkers according
    to the Metropolis algorithm.

    Attributes:
        self.wf_ansatz: the callable wave function ansatz
            signature: (params, x,)
    """

    def __init__(
        self,
        wf_ansatz: Callable[
            [optax_base.Params | dict, jax.Array, np.ndarray], jax.Array
        ],
        **kwargs,
    ) -> None:
        """Init
        Args:
            wf_ansatz: the callable wave function ansatz
            signature: (params, x,)
        """
        self.wf_ansatz = wf_ansatz
        particles = kwargs.get("particles", None)
        masses = kwargs.get("particle_mass", None)
        if particles is not None and masses is not None:
            ms = []
            for particle in particles:
                ms.append([masses[particle]] * 3)
            self.ms = np.array(ms)
            self.move_factor = self.ms
            print(f"Move factor: {self.move_factor},")
        else:
            self.move_factor = 1.0

    def oneshot_sample(
        self,
        xs: jax.Array,
        excitation_number: np.ndarray,
        probability: jax.Array,
        params: jax.Array | np.ndarray | dict,
        step_size: jax.Array,
        key: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array | tuple[jax.Array, ...],
        jax.Array,
    ]:
        """The one-step Metropolis update

        Args:
            xs: (num_of_particles,dim) the configuration cartesian coordinate
                of the particle(s).
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                in the same order as that in coors(flattened).
            probability:(1,) the probability in current particle coordinate
                NOTE: this refers to log probability!
            params: the flow parameter
            step_size: (1,) the step size of each sample step. e.g.
                xs_new = xs + step_size * jax.random.normal(subkey, shape=xs.shape)
            key: the jax PRNG key.

        Returns:
            xs_new: (num_of_particles,dim)the updated xs
            probability_new: (1,)the updated probability
            cond: (1,) the accept condition ,
        """
        key, subkey = jax.random.split(key)
        random_move = (
            step_size
            * jax.random.normal(subkey, shape=xs.shape)
            / jnp.sqrt(self.move_factor)
        )
        xs_new = xs + random_move
        log_wf = self.wf_ansatz(params, xs_new, excitation_number)
        # log probability = log |psi|^2 = 2 Re log |psi|
        probability_new = 2 * log_wf.real
        ratio = jnp.exp(probability_new - probability)
        # Metropolis
        key, subkey = jax.random.split(key)
        cond = jax.random.uniform(subkey, shape=probability.shape) < ratio

        probability_new = jnp.where(cond, probability_new, probability)
        xs_new = jnp.where(cond, xs_new, xs)

        return xs_new, probability_new, cond


@partial(jax.jit, static_argnums=(1,))
def mcmc(
    steps: int,
    metropolis_sampler_batched: Callable,
    key: jax.Array,
    xs_batched: jax.Array,
    excitation_numbers: jax.Array | np.ndarray,
    params: dict,
    probability_batched: jax.Array,
    mc_step_size: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The batched mcmc function for #steps sampling.
    NOTE: this is a jax foriloop implementation

    Args:
        steps: the steps to perform
        metropolis_sampler_batched: the batched metropolis sampler.
        key: the jax.PRNGkey
        xs_batched: (num_of_batch,num_orb,num_of_particles,dim) the batched
            configuration cartesian coordinates
            of the particle(s).
        excitation_numbers: (num_orb, num_of_particles*dim,)
            the corresponding excitation numbers.
        params: the parameters of network
        probability_batched: (num_of_batch,num_orb)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (num_orb,) last mcmc moving step size.
            NOTE: this is a per orbital property!

    Returns:
        key: the jax.PRNGkey
        xs_batched: (num_of_batch,num_orb,num_of_particles,dim) the batched
            configuration cartesian coordinates
            of the particle(s).
        probability_batched: (num_of_batch,num_orb)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (num_orb,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        pmove_per_orb: (num_orb,) the portion of moved particles in last mcmc step.
            NOTE: this is a per orbital property!
    """
    batch_size, num_orb = xs_batched.shape[:2]

    def _body_func(i, val):
        """MCMC Body function"""
        key, xs_batched_old, probability_batched_old, mc_step_size, cond = val
        key, batch_keys = key_batch_split(key, batch_size * num_orb)
        batch_keys = batch_keys.reshape(batch_size, num_orb, 2)
        xs_batched, probability_batched, current_cond = metropolis_sampler_batched(
            xs_batched_old,
            excitation_numbers,
            probability_batched_old,
            params,
            mc_step_size,
            batch_keys,
        )
        # cond: (batch,num_orb,)
        cond += current_cond
        return key, xs_batched, probability_batched, mc_step_size, cond

    mcmc_init_val = (
        key,
        xs_batched,
        probability_batched,
        mc_step_size,
        jnp.zeros((batch_size, num_orb)),
    )
    key, xs_batched, probability_batched, mc_step_size, cond = jax.lax.fori_loop(
        0, steps, _body_func, mcmc_init_val
    )

    # pmove_per_orb: (num_orb,)
    pmove_per_orb = jnp.mean(cond, axis=0) / steps
    # mc_step_size: (num_orb,)
    # TODO: try to adjust step size each 100 update iters!
    # mc_step_size = jnp.where(
    #     pmove_per_orb > 0.90,
    #     mc_step_size * 1.05,
    #     mc_step_size,
    # )
    # mc_step_size = jnp.where(
    #     pmove_per_orb < 0.195,
    #     mc_step_size * 0.905,
    #     mc_step_size,
    # )

    return key, xs_batched, probability_batched, mc_step_size, pmove_per_orb


@partial(
    jax.pmap,
    axis_name="xla_device",
    in_axes=(None, 0, 0, None, None, 0, None, None),
    out_axes=(0, 0, 0, None, None),
    static_broadcasted_argnums=(7,),
)
def mcmc_pmap_ebes(
    steps: int,
    key: jax.Array,
    xs_batched: jax.Array,
    excitation_number: jax.Array | np.ndarray,
    params: dict,
    probability_batched: jax.Array,
    mc_step_size: jax.Array,
    wf_ansatz: Callable[[optax_base.Params | dict, jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The pmaped mcmc function (ebes) for #steps sampling.

    Args:
        steps: the steps to perform
        key: (num_of_xla_device,) the jax.PRNGkey
        xs_batched: (num_of_xla_device,batch_per_device,num_of_particles,dim)
            the batched
            configuration cartesian coordinates
            of the particle(s).
        params: the parameters of network
        probability_batched: (num_of_xla_device,batch_per_device,)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (1,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        wf_ansatz: the callable wave function ansatz
            signature: (params, x,)

    Returns:
        key: (num_of_xla_device,) the jax.PRNGkey
        xs_batched: (num_of_xla_device,batch_per_device,num_of_particles,dim)
            the batched
            configuration cartesian coordinates
            of the particle(s).
        probability_batched: (num_of_xla_device,batch_per_device,)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (1,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        pmove_per_orb: (1,) the portion of moved particles in last mcmc step.
            NOTE: this is a per orbital property!
    """
    key, key_mcmc = jax.random.split(key, 2)

    def _logp(x):
        """Log probability on SINLGE configuration x
        Args:
            x: (num_of_particles,dim)
        """
        return 2 * wf_ansatz(params, x, excitation_number).real

    logp_batched = jax.vmap(_logp)

    xs_batched, probability_batched, key, accept_rate, recip_ratio = mcmc_ebes_kernal(
        logp_fn=logp_batched,
        x_init=xs_batched,
        key=key_mcmc,
        mc_steps=steps,
        mc_width=mc_step_size,
        logp_init=probability_batched,
    )

    # pmove_per_orb: (1,)
    pmove_per_orb = jax.lax.pmean(accept_rate, axis_name="xla_device")
    # mc_step_size: (1,)

    # TODO: try to adjust step size each 100 update iters!
    # mc_step_size = jnp.where(
    #     pmove_per_orb > 0.99,
    #     mc_step_size * 1.05,
    #     mc_step_size,
    # )
    # mc_step_size = jnp.where(
    #     pmove_per_orb < 0.095,
    #     mc_step_size * 0.905,
    #     mc_step_size,
    # )

    return key, xs_batched, probability_batched, mc_step_size, pmove_per_orb


@partial(
    jax.pmap,
    axis_name="xla_device",
    in_axes=(None, 0, 0, None, None, 0, None, None),
    out_axes=(0, 0, 0, None, None),
    static_broadcasted_argnums=(7,),
)
def mcmc_pmap(
    steps: int,
    key: jax.Array,
    xs_batched: jax.Array,
    excitation_numbers: jax.Array | np.ndarray,
    params: dict,
    probability_batched: jax.Array,
    mc_step_size: jax.Array,
    metropolis_sampler_batched: Callable,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The pmaped mcmc function (Traditional) for #steps sampling.

    Args:
        steps: the steps to perform mcmc move
        key: (num_of_xla_device,) the jax.PRNGkey
        xs_batched: (num_of_xla_device,batch_per_device,num_orb,num_of_particles,dim)
            the batched
            configuration cartesian coordinates
            of the particle(s).
        excitation_numbers: (num_orb, num_of_particles*dim,) the
            corresponding excitation numbers.
        params: the parameters of network
        probability_batched: (num_of_xla_device,batch_per_device,num_orb,)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (num_orb,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        metropolis_sampler_batched: the batched metropolis sampler.

    Returns:
        key: (num_of_xla_device,) the jax.PRNGkey
        xs_batched: (num_of_xla_device,batch_per_device,num_orb,num_of_particles,dim)
            the batched
            configuration cartesian coordinates
            of the particle(s).
        probability_batched: (num_of_xla_device,batch_per_device,num_orb)
            the batched probability
                NOTE:this refers to log probability!
        mc_step_size: (num_orb,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        pmove_per_orb: (num_orb,) the portion of moved particles in last mcmc step.
            NOTE: this is a per orbital property!
    """
    key, key_mcmc = jax.random.split(key, 2)

    key, xs_batched, probability_batched, mc_step_size, pmove_per_orb = mcmc(
        steps=steps,
        metropolis_sampler_batched=metropolis_sampler_batched,
        key=key_mcmc,
        xs_batched=xs_batched,
        excitation_numbers=excitation_numbers,
        params=params,
        probability_batched=probability_batched,
        mc_step_size=mc_step_size,
    )
    return key, xs_batched, probability_batched, mc_step_size, pmove_per_orb

"""Wave Function Basis"""

from functools import partial

from typing import Callable
import warnings
import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.utils.init_molecule import InitMolecule


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def hermite(n: int, x: float) -> jax.Array:
    """The (coefficient included) hermite polynominals
    NOTE: if one would like to denote the `pure` Hermite polynominals as
        Hn(x) for example H0(x) = 1, H1(x) = 2x, H2(x) = 4x^2 - 2, ...
        aka, a physicist's hermite polynominals
        (see https://en.wikipedia.org/wiki/Hermite_polynomials)
    Then the hermite defined HERE is the pure hermite multiplied with
        some coefficients to make an easier implementation of the
        recurrence relation. And hence this hermite shoule
        ONLY be used in a 1D Harmonic Oscillator case!

    Args:
        n: the order of the hermite polynominal
        x: the variable of hermite polynominal
    Returns:
        the hermite polynominal of order n at x which is multiplied
            with the coefficients for a 1D Harmonic Oscillator, resulting in
            hn(x) = 1/sqrt(2^n*n!) * (1/pi)^(1/4) * Hn(x)

    """
    h0 = 1.0 / jnp.pi ** (1 / 4)
    h1 = jnp.sqrt(2.0) * x / jnp.pi ** (1 / 4)

    def body_fun(i, val):
        """Body function for recurrence relation
        Args:
            i: the current order of the hermite polynominal
            val: the previous two values of the hermite polynominal
                hi_2: h_{i-2}
                hi_1: h_{i-1}
        Returns:
            hi_1: h_{i-1}
            hi: the i-th hermite polynominal
        """
        hi_2, hi_1 = val
        return hi_1, jnp.sqrt(2.0 / i) * x * hi_1 - jnp.sqrt((i - 1) / i) * hi_2

    _, hn = jax.lax.fori_loop(2, n + 1, body_fun, (h0, h1))

    return jax.lax.cond(n > 0, lambda: hn, lambda: h0)


@hermite.defjvp
def hermite_jvp(n: int, primals, tangents):
    """Custom derivative of (coefficient included) hermite polynominals"""
    (x,) = primals
    (dx,) = tangents
    hn = hermite(n, x)
    dhn = jnp.sqrt(2 * n) * hermite((n - 1) * (n > 0), x) * dx
    primals_out, tangents_out = hn, dhn
    return primals_out, tangents_out


def log_wf_base_1d(x: float, c: float, m: float, w: float, n: int) -> jax.Array:
    """The wave function of 1D Harmonic Oscillator.
    psi_n(x)
    = (1/sqrt(2^n*n!)) * (m*w/(hbar*pi))^(1/4)
        * e^(-m*w*(x-c)^2/(2hbar)) * Hn(sqrt(m*w/hbar)(x-c))

    NOTE: the `hermite` here is with coefficients like
        (1/sqrt(2^n*n!)) * (1/pi)^(1/4)
        hence here log_psi doesn't contain those n-relative
        coefficients.
    NOTE: 1D! The eigenstates of a one-dimensional
    harmonic oscillator with frequency w, centered at x=c.

    Args:
        x: the 1D coordinate of the (single) coordinate.
        c: the center of the Hermite function.
        m: the mass of the particle in a.u.
        w: the frequency of the oscillator in a.u.
        n: the excitation quantum number

        NOTE: n=0 for GS!

    Returns:
        log_psi: the log probability amplitude at x.
        NOTE: this is log|psi|
    """

    log_psi = (
        jnp.log(m * w) / 4
        - 0.5 * m * w * (x - c) ** 2
        + jnp.log(jnp.abs(hermite(n, jnp.sqrt(m * w) * (x - c))))
    )
    return log_psi


class HermiteFunction:
    """Harmonic Oscillator Wavefunction Centered at equilibrium configuration.
    (Hermite Functions, which is often called Hermite-Gaussian functions)
    """

    def __init__(
        self,
        mole_init_obj: InitMolecule,
    ) -> None:
        """Init
        Args:
            particles: the tuple with atoms names, for example,
                ("C","H","H","H","H","H")
            m: the mass of the particle in a.u.
                with key as the atom name and value as the mass
            w: the frequency of each particle in a.u.
                with key as the atom name and value as the frequency
            x0: (num_particles*dim,) the flattened reference point
                of wavefunc basis in latent space.
        """
        self.particles = mole_init_obj.particles
        self.m = mole_init_obj.particle_mass
        self.w = mole_init_obj.omega_for_wf_basis
        self.x0 = np.zeros_like(mole_init_obj.eq_config.reshape(-1))
        print("\nWavefunction basis reference: " f": {self.x0}")

    @property
    def log_phi_base(self) -> Callable:
        """The log phi base
        That would be used in wavefunction ansatze.

        Returns:
            _log_phi_base: the log wavefunction base function
        """
        _log_phi_base = self.log_hermite_func
        return _log_phi_base

    def log_hermite_func(
        self,
        coors: jax.Array | np.ndarray,
        excitation_number: np.ndarray,
    ) -> jax.Array:
        """the hermite functions that centered at equilibrium configuration
        of the particle.

        Args:
            coors: (num_particles,dim) the full configuration coordinates
                of the system, ordered as in __init__, `particles`.
            excitation_number: (num_particles*dim,) the corresponding excitation
                quantum number of each 1d-oscillator (of each 1d coordinate),
                 in the same order as that in coors(flattened).
        Returns:
            phi_base: the full hermite function wavefunction
                phi_base = log_hermite1 + log_hermite2 + ...
        """
        num_particles, dim = coors.shape
        coors = coors.reshape(-1)
        ms = []
        ws = []
        for particle in self.particles:
            ms.append([self.m[particle]] * dim)
            ws.append([self.w[particle]] * dim)
        ms = np.array(ms).reshape(-1)
        ws = np.array(ws).reshape(-1)

        # print(
        #     f"ms={ms}\nws={ws}\ncoors={coors}\nexcitation_number={excitation_number}"
        # )

        phis = jax.vmap(log_wf_base_1d, in_axes=(0, 0, 0, 0, 0))(
            coors, self.x0, ms, ws, excitation_number
        )
        phi_base = jnp.sum(phis)
        return phi_base


def log_gaussian_1d(
    x: jax.Array | np.ndarray, x0: jax.Array | np.ndarray, sigma: float = 1.0
) -> jax.Array:
    """The 1D Gaussian basis
    centered at x0, normalized
    in log domain

    Args:
        x: (1,) the coordinate of one degree of freedom
        x0: (1,) the center of the gaussian
        sigma: the standard deviance of the gaussian.
    Returns:
        gaussian_base: the single gaussian base
            normalized wavefunction
            in log domain
            (-(x-x0)^2/(2 sigma^2))-(1/2)log(2 pi sigma^2)
    """
    r = jnp.linalg.norm(x - x0)
    log_gaussian_base = (-(r**2) / (2 * sigma**2)) - (1 / 2) * jnp.log(
        2 * jnp.pi * sigma**2
    )
    return log_gaussian_base


def gaussian(
    x: jax.Array | np.ndarray, x0: jax.Array | np.ndarray, sigma: float = 1.0
) -> jax.Array:
    """The 3D Gaussian basis
    centered at x0, normalized

    Args:
        x: (3,) the coordinate of the particle
        x0: (3,) the center of the gaussian
        sigma: the standard deviance of the gaussian.
    Returns:
        gaussian_base: the single gaussian base
            normalized wavefunction
            exp(-(x-x0)^2/(2 sigma^2))/(2 pi sigma^2)^(3/2)
    """
    r = jnp.linalg.norm(x - x0)
    gaussian_base = jnp.exp(-(r**2) / (2 * sigma**2)) / (
        2 * jnp.pi * sigma**2
    ) ** (3 / 2)
    return gaussian_base


def log_gaussian(
    x: jax.Array | np.ndarray, x0: jax.Array | np.ndarray, sigma: float = 1.0
) -> jax.Array:
    """The 3D Gaussian basis
    centered at x0, normalized
    in log domain

    Args:
        x: (3,) the coordinate of the particle
        x0: (3,) the center of the gaussian
        sigma: the standard deviance of the gaussian.
    Returns:
        gaussian_base: the single gaussian base
            normalized wavefunction
            in log domain
            (-(x-x0)^2/(2 sigma^2))-(3/2)log(2 pi sigma^2)
    """
    r = jnp.linalg.norm(x - x0)
    log_gaussian_base = (-(r**2) / (2 * sigma**2)) - (3 / 2) * jnp.log(
        2 * jnp.pi * sigma**2
    )
    return log_gaussian_base


def log_gaussian_mixture(
    x: jax.Array | np.ndarray, x0s: jax.Array | np.ndarray, sigma: float = 1.0
) -> jax.Array:
    """The 3D Gaussian basis
    A mixture of different Gaussians centered
    at each of x0s
    in log domain

    Args:
        x: (3,) the coordinate of the particle
        x0s: (num_of_gaussians,3) the center of the different gaussians
        sigma: the standard deviance of the gaussian.
    Returns:
        gaussian_base: the single gaussian base
            normalized wavefunction
            in log domain

            sum_i pi_i exp(-(x-x0_i)^2/(2 sigma^2))/(2 pi sigma^2)^(3/2)
            with sum_i pi_i = 1
    """
    num_of_gaussian = x0s.shape[0]
    weight = 1 / num_of_gaussian
    gaussians = jax.vmap(gaussian, in_axes=(None, 0, None))(x, x0s, sigma)
    gaussians *= weight
    log_gaussian_base = jnp.log(jnp.sum(gaussians))
    return log_gaussian_base


class InvariantGaussian:
    """Gaussian which is permutation invariant with hydrogen

    Attributes:
        self.x0s: (num_particles,dim) the corresponding x0 of each particle
        self.sigmas:(num_particles,) the corresponding sigmas of each
                gaussian. If None, then initialized as 1.0
        self.log_phi_base

    """

    def __init__(
        self,
        particles: tuple,
        partition: np.ndarray | list[int],
        sigmas: jax.Array | np.ndarray | None = None,
        symmetry_implement: str = "simple",
        symmetry_radius: float | None = 1.0,
    ) -> None:
        """Init invaraint gaussian

        Args:
            particles: the tuple with atoms names, for example,
                ("C","H","H","H","H","H")
            partition: the array denoting the permutational
                equivalent part, the equivalent parts
                are seperated by the indices of the partition
                array.
                for example, if particles are [C H H H H H] and
                partition=[1] then the first particle
                is not permutative with any other particles
                while all the other particles counting from
                1 to the last one are all permutatively equivalent.
            sigmas: (num_particles,) the corresponding sigmas of each
                gaussian. If None, then initialized as 1.0
            symmetry_implement: avaliable: `simple`, `use_partition`
                the method used to implement symmetry
                feature. If `simple` then partition would not be
                used, and all the gaussians are initialized around
                the origin of the coordinates. If `use_partition`
                is used then initialize gaussians as stated in partitions.
            symmetry_radius: the radius of the circle w.r.t which
                the gaussians would be symmetrically placed.

        NOTE: Since that when setting all the gaussians with x0=0
            is exactly a symmetry distribution w.r.t the origin,
            the symmetry_implement `simple` is exactly what we want.
            TODO: remove symmetry_implement choice and only use `simple`
        """
        partition = np.array(partition)
        if len(partition.shape) > 1:
            raise NotImplementedError(
                "Partition with more than 2 equivalent group" " is not supported!"
            )
        num_of_particles = len(particles)
        num_of_equl_particles = num_of_particles - partition[0]

        if symmetry_implement == "use_partition":
            raise NotImplementedError(
                "If not using GMM(Gaussian Mixture Model)"
                " then partition could not be used!"
                "\nPlease make sure that currently call"
                " is based on GMM but not a single Gaussian!"
            )
            x0s = []

            for i in range(partition[0]):
                x0s.append([0.0, 0.0, 0.0])

            for i in range(num_of_equl_particles):
                theta = (i + 1) * 2 * np.pi / num_of_equl_particles
                xi = 0.0
                yi = symmetry_radius * np.sin(theta)
                zi = symmetry_radius * np.cos(theta)
                x0s.append([xi, yi, zi])

            self.x0s = np.array(x0s)
        elif symmetry_implement == "simple":
            warnings.warn(
                "When symmetry_implement is set to `simple`, "
                f"desinated partition={partition}"
                " will not be used."
            )
            self.x0s = np.zeros(num_of_particles * 3)
            # self.x0s = equilibrium_bowman_jpca_2006_110_1569_1574()[3::].reshape(
            #     (num_of_particles, 3)
            # )

        self.sigmas = sigmas if sigmas is not None else np.ones(num_of_particles * 3)

        print(f"particles={particles}\npartition={partition}\nx0s={self.x0s}")
        print(f"sigmas={self.sigmas}")

    @property
    def log_phi_base(self) -> callable:
        """The log phi base
        That would be used in wavefunction ansatze.

        Returns:
            _log_phi_base: the log wavefunction base function
        """
        _log_phi_base = self.log_invariant_gaussian
        # _log_phi_base = self.log_invariant_gaussian_mixture
        return _log_phi_base

    def log_invariant_gaussian(self, coors: jax.Array | np.ndarray) -> jax.Array:
        """The permutational invariant gaussian basis

        Args:
            coors: (num_particles,dim) the full configuration coordinates
                of the system, ordered as in __init__, `particles`.


        Returns:
            phi_base: the full invariant gaussian basis wavefunction
                phi_base = log_gaussian1 + log_gaussian2 + ...
        """
        coors = coors.reshape(-1)
        phis = jax.vmap(log_gaussian_1d)(coors, self.x0s, self.sigmas)
        phi_base = jnp.sum(phis)
        return phi_base

    def log_invariant_gaussian_mixture(
        self, coors: jax.Array | np.ndarray
    ) -> jax.Array:
        """The permutational invariant gaussian basis
        implemented by gaussian mixture model (GMM)

        Args:
            coors: (num_particles,dim) the full configuration coordinates
                of the system, ordered as in __init__, `particles`.


        Returns:
            phi_base: the full invariant gaussian basis wavefunction
                phi_base = log_gaussian1 + log_gaussian2 + ...
        """
        raise NotImplementedError(
            "WF basis for 15 multiplication of 1d gaussians not implemented!"
        )
        phis = jax.vmap(log_gaussian_mixture, in_axes=(0, None, 0))(
            coors, self.x0s, self.sigmas
        )
        phi_base = jnp.sum(phis)
        return phi_base

"""The Jax Call Backed JBB PES Function"""

import numpy as np
import jax
import jax.numpy as jnp

try:
    from neuralvib.molecule.ch5plus.JBB_Full_PES import JBBCH5ppotential
except ImportError:
    print("ImportError: JBBCH5ppotential not found. Please check the import path.")

jax.config.update("jax_enable_x64", True)


def jbbf2py_pes_polar(r):
    """The ch5ppot_func in polar coordinates
    Return in cm-1
    """
    return JBBCH5ppotential.ch5ppot_func(r)


@jax.jit
def jbbjax_polar(r: jax.Array) -> float:
    """The pure called back jax function
    of the ch5ppot_func in polar coordinates,
    return in cm-1
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(jbbf2py_pes_polar, result_shape, r)


def jbbf2py_pes_cartesian(xn: jax.Array | np.ndarray) -> float:
    """The pes in cartesian coordinates.

    Args:
        xn: (3,6)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,6)!
            see getpot.f for details.

    Returns:
        epot: the potential energy in a.u.
        NOTE: The direct value returned by getpot
            is added by De. Hence to get the pes
            which has 0 energy at equilibrium
            configuration, one needs to subtract
            De manually.
    """
    de = -40.6527648702729
    epot_direct = JBBCH5ppotential.getpot_with_return(xn)
    epot = epot_direct - de

    # reject if PES give unphysical negative energy
    if epot < 0:
        # return 0.09112669999999999
        return 10.0

    return epot


def jbbexternal_withreject(xn: np.ndarray) -> float:
    """The exteranl pes function

    Args:
        xn: (3,6)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,6)!
            see getpot.f for details.

    Returns:
        epot: the potential energy in a.u.
        NOTE: The direct value returned by getpot
            is added by De. Hence to get the pes
            which has 0 energy at equilibrium
            configuration, one needs to subtract
            De manually.
    """
    vceil = 0.09112669999999999  # 20000 cm-1

    hydrogens = xn.T[1::]
    rij = hydrogens[:, None, :] - hydrogens
    r = np.linalg.norm(rij, axis=-1)
    triu_ind = np.triu_indices_from(r, k=1)
    r_up = r[triu_ind]

    # Reject as Carrington's code
    distmin = 1.12
    dist2 = 1.70
    dist3 = 1.90
    dist4 = 3.0

    if (r_up < distmin).sum() >= 1:
        return vceil
    if (r_up < dist2).sum() >= 2:
        return vceil
    if (r_up < dist3).sum() >= 3:
        return vceil
    if (r_up < dist4).sum() >= 4:
        return vceil

    epot = jbbf2py_pes_cartesian(xn=xn)
    return epot


def jbbexternal(xn: np.ndarray) -> float:
    """The exteranl pes function

    Args:
        xn: (3,6)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,6)!
            see getpot.f for details.

    Returns:
        epot: the potential energy in a.u.
        NOTE: The direct value returned by getpot
            is added by De. Hence to get the pes
            which has 0 energy at equilibrium
            configuration, one needs to subtract
            De manually.
    """
    epot = jbbf2py_pes_cartesian(xn=xn)
    return epot


def jbbf2py_pes_ch5ppot_cart(cartr: jax.Array | np.ndarray) -> float:
    """The pes in cartesian coordinates.

    Args:
        cartr: (3,5)  is the Cartesian coordinates for
        FIVE atoms in order of
        H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,5)!
            see ch5ppot.dist6.bond.f90 for details.

    Returns:
        epot: the potential energy in a.u.
    """
    epot_direct = JBBCH5ppotential.ch5ppot_func_cart(cartr)
    return epot_direct


@jax.jit
def jbbjax_ch5ppot_cart(cartr: jax.Array) -> float:
    """The pure called back jax function

    Args:
        cartr: (3,5)  is the Cartesian coordinates for
        FIVE atoms in order of
        H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,5)!
            see ch5ppot.dist6.bond.f90 for details.

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(jbbf2py_pes_ch5ppot_cart, result_shape, cartr)



# -------- JAX wrapper: vmap-friendly via vmap_method="expand_dims" --------
def jbbjax_cartesian(xb: jax.Array) -> jax.Array:
    """
    Evaluate the CH5⁺ potential via a host (f2py) callback.
    NOTE: OpenMP parallel version!

    Accepts any leading batch shape: (..., 3, 6) with atoms ordered C, H, H, H, H, H (bohr).
    Works under `jit`, `vmap`, and nested `vmap`s. Parallelism is provided by OpenMP in
    the Fortran `getpot_batch` routine, which is called once per JAX host callback.

    Args:
        xb: jax.Array with shape (..., 3, 6), dtype float64 (will be cast as needed).

    Returns:
        jax.Array with shape (...,), dtype float64. Energies in a.u. (after optional `DE` shift).
    """
    xb = jnp.asarray(xb, dtype=jnp.float64)
    if xb.shape[-2:] != (3, 6):
        raise ValueError(f"Expected xb to have trailing shape (3,6), got {xb.shape}")
    out_shape = xb.shape[:-2]  # preserve all leading batch dims
    out_spec = jax.ShapeDtypeStruct(out_shape, jnp.float64)

    # Important: make pure_callback vmap-compatible
    return jax.pure_callback(
        _f2py_getpot_batch,
        out_spec,
        xb,
        vmap_method="expand_dims",   # <- fixes your NotImplementedError under vmap
    )


# Constants (cm^-1 -> hartree)
CM_TO_HARTREE = 1.0 / 219474.6313705
V_NEG  = 20000000.0 * CM_TO_HARTREE
DE = np.float64(-40.6527648702729)  # or your chosen correction

def _f2py_getpot_batch(xb_np: np.ndarray) -> np.ndarray:
    """
    Inputs:
      xb_np: (..., 3, 6) Cartesian coords in bohr (atom 0 = C, atoms 1..5 = H)
    Returns:
      potentials in hartree, guards applied.
    """
    xb_np = np.asarray(xb_np, dtype=np.float64)
    assert xb_np.shape[-2:] == (3, 6)

    lead = xb_np.shape[:-2]
    B = int(np.prod(lead)) if lead else 1

    # Flatten for F2PY: (B,3,6) -> Fortran (3,6,B)
    xb_flat = xb_np.reshape((B, 3, 6))
    xb_f = np.asfortranarray(np.transpose(xb_flat, (1, 2, 0)))

    # Raw potentials from Fortran (hartree)
    epots = np.asarray(JBBCH5ppotential.getpot_batch(xb_f, np.int32(B)),
                       dtype=np.float64).reshape(B)

    epots -= DE

    mask_neg = epots < 0.0
    epots[mask_neg] = V_NEG

    # Reshape back to leading dims
    epots = epots.reshape(lead)

    return epots

"""The Jax Call Backed JBB PES Function"""

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize

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


def jbbjax_polar_vec(r: jax.Array) -> float:
    """The pure called back jax function"""
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(jbbf2py_pes_polar, result_shape, r, vectorized=True)


# def manual_hessian():
#     """Manually calculate Hessian"""
#     # init_guess = np.random.uniform(0,0.00000001,18)
#     init_guess = np.zeros(18)
#     def _potential_mwcd(mwcd:np.ndarray) -> float:
#         input2jbb = massweight2polar(mwcd)
#         return jbbf2py_pes_polar(input2jbb)
#     res = scipy.optimize.minimize(_potential_mwcd,init_guess,method="BFGS")
#     return res


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


def _jbbjax_cartesian(xn: jax.Array) -> float:
    """The pure called back jax function

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
    # reject artifacts in PES
    # See Wang and Carrington, J. Chem. Phys. 129, 234102(2008)
    # vceil = 0.05467602 # 12000 cm-1
    vceil = 0.09112669999999999  # 20000 cm-1

    hydrogens = xn.T[1::]
    rij = hydrogens[:, None, :] - hydrogens
    r = jnp.linalg.norm(rij, axis=-1)
    triu_ind = jnp.triu_indices_from(r, k=1)
    r_up = r[triu_ind]

    # Reject as Carrington's code
    distmin = 1.12
    dist2 = 1.70
    dist3 = 1.90
    dist4 = 3.0
    smallerthanmin = (r_up < distmin).sum()
    smallerthan2 = (r_up < dist2).sum()
    smallerthan3 = (r_up < dist3).sum()
    smallerthan4 = (r_up < dist4).sum()
    rejectmin = jnp.where(smallerthanmin >= 1, True, False)
    reject2 = jnp.where(smallerthan2 >= 2, True, False)
    reject3 = jnp.where(smallerthan3 >= 3, True, False)
    reject4 = jnp.where(smallerthan4 >= 4, True, False)
    # Reject as Carrington's code end.

    # DIY Reject
    # distmin = 1.4
    # dist2 = 1.70
    # dist3 = 1.90
    # dist4 = 3.2
    # smallerthanmin = (r_up < distmin).sum()
    # smallerthan2 = (r_up < dist2).sum()
    # smallerthan3 = (r_up < dist3).sum()
    # smallerthan4 = (r_up < dist4).sum()
    # rejectmin = jnp.where(smallerthanmin >= 1, True, False)
    # reject2 = jnp.where(smallerthan2 >= 2, True, False)
    # reject3 = jnp.where(smallerthan3 >= 2, True, False)
    # reject4 = jnp.where(smallerthan4 >= 3, True, False)
    # DIY Reject End

    rejection_array = jnp.array([rejectmin, reject2, reject3, reject4])
    reject = jax.lax.select(
        (rejection_array.sum() > 0),
        True,
        False,
    )

    # reject = jnp.where(smallerthanmin >= 2 or smallerthan5 >= 5, True, False)
    # if not reject, return direct potential energy
    # from JBB Full PES
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(jbbf2py_pes_cartesian, result_shape, xn)

    result = jnp.where(reject == True, vceil, epot)

    return result


def jbbjax_cartesian(xn: jax.Array) -> float:
    """The pure called back jax function
    without reject

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
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(jbbf2py_pes_cartesian, result_shape, xn,vmap_method="sequential")
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

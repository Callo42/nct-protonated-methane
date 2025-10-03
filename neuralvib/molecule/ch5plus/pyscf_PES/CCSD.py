"""Hartree-Fock PES from PySCF"""
# %%
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import cc


def pyscf_ccsd_pes_cartesian(
    cartesian_coors: jax.Array | np.ndarray, basis: str = "cc-pVTZ"
) -> float:
    """The pyscf pes in cartesian coordinates.
    Coupled Cluster Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        basis: the basis to use

    Returns:
        epot: the potential energy in a.u.
        NOTE: the pes only returns the energy DIFFERENCE between the
            global minimum and the current configuration!
            Currently the global minimum V0 needs to be manually
            set here.
    """
    if basis == "cc-pVTZ":
        V0 = -40.66081206718982  # CCSD/cc-PVTZ
    elif basis == "cc-pVQZ":
        V0 = -40.68588434795463  # CCSD/cc-PVQZ
    elif basis == "aug-cc-pVTZ":
        V0 = -40.66552647806463  # CCSD/aug-cc-pVTZ
    else:
        raise NotImplementedError(f"V0 for basis {basis} not implemented!")
    atom_list = ["C", "H", "H", "H", "H", "H"]

    conf = cartesian_coors
    mol = gto.Mole()
    mol.unit = "B"
    mol.atom = []

    for i, atom in enumerate(atom_list):
        mol.atom.append([atom, tuple(conf[i])])

    mol.charge = +1
    mol.basis = basis
    mol.build()
    mol.verbose = False

    mf = scf.HF(mol).run()
    mf_cc = cc.CCSD(mf).run()
    epot = mf_cc.e_tot - V0
    return epot


def pyscf_ccsd_jax_pure_cartesian(cartesian_coors: jax.Array) -> float:
    """The pure called back jax function
    Coupled Cluster Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(pyscf_ccsd_pes_cartesian, result_shape, cartesian_coors)

    return epot


def pyscf_ccsd_jax_io_cartesian(cartesian_coors: jax.Array) -> float:
    """The io called back jax function
    Coupled Cluster Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.experimental.io_callback(
        pyscf_ccsd_pes_cartesian, result_shape, cartesian_coors
    )

    return epot

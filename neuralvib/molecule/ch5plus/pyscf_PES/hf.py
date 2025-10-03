"""Hartree-Fock PES from PySCF"""

# %%
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from pyscf import scf


def pyscf_hf_pes_cartesian(
    cartesian_coors: jax.Array | np.ndarray, basis: str = "cc-pVTZ"
) -> float:
    """The pyscf pes in cartesian coordinates.
    Hartree Fock Theory

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
        V0 = -40.4210362637797  # HF/cc-PVTZ
    elif basis == "cc-pVQZ":
        V0 = -40.4240285224631  # HF/cc-PVQZ
    elif basis == "aug-cc-pVTZ":
        V0 = -40.4214147785129  # HF/aug-cc-pVTZ
    else:
        raise NotImplementedError(f"V0 for basis {basis} not implemented!")

    atom_list = ["C", "H", "H", "H", "H", "H"]

    mol = gto.Mole()
    mol.unit = "B"

    atom_array = np.array(atom_list)
    atom_input = list(zip(atom_array, tuple(cartesian_coors)))
    mol.atom = atom_input

    mol.charge = +1
    mol.basis = basis
    mol.build()
    mol.verbose = False

    mf = scf.HF(mol)
    mf.kernel()
    epot = mf.e_tot - V0
    return epot


@jax.jit
def pyscf_hf_jax_pure_cartesian(cartesian_coors: jax.Array) -> float:
    """The pure called back jax function
    Hartree Fock Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(pyscf_hf_pes_cartesian, result_shape, cartesian_coors)

    return epot


@jax.jit
def pyscf_hf_jax_io_cartesian(cartesian_coors: jax.Array) -> float:
    """The io called back jax function
    Hartree Fock Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.experimental.io_callback(
        pyscf_hf_pes_cartesian, result_shape, cartesian_coors, ordered=False
    )

    return epot

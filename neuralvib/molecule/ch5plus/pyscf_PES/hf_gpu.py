"""Hartree-Fock PES from PySCF"""
# %%
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from gpu4pyscf import scf


def gpu4pyscf_hf_pes_cartesian(cartesian_coors: jax.Array | np.ndarray) -> float:
    """The gpu4pyscf pes in cartesian coordinates.
    Hartree Fock Theory

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
        NOTE: the pes only returns the energy DIFFERENCE between the
            global minimum and the current configuration!
            Currently the global minimum V0 needs to be manually
            set here.
    """
    print("Running gpu4pyscf")
    V0 = -40.4210362637797  # HF/cc-PVTZ
    atom_list = ["C", "H", "H", "H", "H", "H"]
    basis = "cc-pVTZ"

    mol = gto.Mole()
    mol.unit = "B"
    # mol.atom = []
    # mol.verbose = False

    # for i, atom in enumerate(atom_list):
    #     mol.atom.append([atom, tuple(conf[i])])

    atom_array = np.array(atom_list)
    atom_input = list(zip(atom_array, tuple(cartesian_coors)))
    mol.atom = atom_input

    mol.charge = +1
    mol.basis = basis
    mol.build()

    mf = scf.HF(mol)
    mf.kernel()
    epot = mf.e_tot - V0
    return epot


@jax.jit
def gpu4pyscf_hf_jax_pure_cartesian(cartesian_coors: jax.Array) -> float:
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
    epot = jax.pure_callback(gpu4pyscf_hf_pes_cartesian, result_shape, cartesian_coors)

    return epot


@jax.jit
def gpu4pyscf_hf_jax_io_cartesian(cartesian_coors: jax.Array) -> float:
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
        gpu4pyscf_hf_pes_cartesian, result_shape, cartesian_coors
    )

    return epot

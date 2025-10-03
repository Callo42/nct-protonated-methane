"""MP2 PES from PySCF"""

# %%
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from pyscf import scf


def pyscf_mp2_pes_cartesian(
    cartesian_coors: jax.Array | np.ndarray, basis: str = "cc-pVTZ"
) -> float:
    """The pyscf pes in cartesian coordinates.
    MP2

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
        V0 = -40.63954865639959  # MP2/cc-pVTZ
    elif basis == "cc-pVQZ":
        V0 = -40.63954865639959  # MP2/cc-PVQZ
    elif basis == "aug-cc-pVTZ":
        V0 = -40.6442538662981  # MP2/aug-cc-pVTZ
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

    try:
        mf = scf.HF(mol).run()
        mf = mf.MP2().run()
    except Exception as e:
        epot = 100.0
    else:
        epot = mf.e_tot - V0
    return epot


def pyscf_mp2_jax_pure_cartesian(cartesian_coors: jax.Array) -> float:
    """The pure called back jax function
    MP2

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(pyscf_mp2_pes_cartesian, result_shape, cartesian_coors)

    return epot


def pyscf_mp2_jax_io_cartesian(cartesian_coors: jax.Array) -> float:
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
        pyscf_mp2_pes_cartesian, result_shape, cartesian_coors
    )

    return epot

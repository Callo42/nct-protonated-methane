"""Hartree-Fock PES from PySCF"""

# %%
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from pyscf import scf
from pyscf import gto
import pyscf
from pyscf import dft


def pyscf_dft_pes_cartesian(
    cartesian_coors: jax.Array | np.ndarray, basis: str = "cc-pVTZ", xc: str = "M05"
) -> float:
    """The pyscf pes in cartesian coordinates.
    DFT

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        basis: the basis to use
        xc: the exchange-correlation functional

    Returns:
        epot: the potential energy in a.u.
        NOTE: the pes only returns the energy DIFFERENCE between the
            global minimum and the current configuration!
            Currently the global minimum V0 needs to be manually
            set here.
    """
    if basis == "6-31g" and xc == "CAMB3LYP":
        V0 = -40.6717234563847  # DFT/6-31g-CAMB3LYP
    elif basis == "cc-pVTZ" and xc == "CAMB3LYP":
        V0 = -40.71465724047103  # DFT/cc-pVTZ-CAMB3LYP
    elif basis == "cc-pVTZ" and xc == "M05":
        V0 = -40.70667958286832  # DFT/cc-pVTZ-M05
    elif basis == "cc-pVQZ" and xc == "M05":
        V0 = -40.7110022044054  # DFT/cc-pVQZ-M05
    elif basis == "aug-cc-pVTZ" and xc == "M05":
        V0 = -40.707041734035  # DFT/aug-cc-pVTZ-M05
    else:
        raise NotImplementedError(f"V0 for basis {basis} and xc {xc} not implemented!")

    atom_list = ["C", "H", "H", "H", "H", "H"]
    atom_array = np.array(atom_list)
    atom_input = list(zip(atom_array, tuple(cartesian_coors)))

    mol = gto.M(
        atom=atom_input,
        unit="B",
        charge=+1,
        basis=basis,
    )

    try:
        mol.verbose = False
        mf = dft.KS(mol)
        mf.xc = xc
        mf.density_fit()
        mf.kernel()
    except Exception as e:
        epot = 1000.0
    else:
        epot = mf.e_tot - V0
    return epot


def pyscf_dft_jax_pure_cartesian(cartesian_coors: jax.Array) -> float:
    """The pure called back jax function
    dft

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.pure_callback(pyscf_dft_pes_cartesian, result_shape, cartesian_coors)

    return epot


def pyscf_dft_jax_io_cartesian(cartesian_coors: jax.Array) -> float:
    """The io called back jax function
    DFT

    Args:
        cartesian_coors: (6,3)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)

    Returns:
        epot: the potential energy in a.u.
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    epot = jax.experimental.io_callback(
        pyscf_dft_pes_cartesian, result_shape, cartesian_coors
    )

    return epot

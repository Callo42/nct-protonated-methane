"""Unit conversion relationships"""
from typing import Union
import numpy as np


def _convert_Hartree_to_inverse_cm(energy_in_Hartree: float) -> float:
    """Convert energy from Hartree (E_h)
    to wavenumber cm^{-1}

    Args:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h

    Returns:
        energy_in_inverse_cm: the corresponding energy in
            unit of wavenumber cm^{-1}
    """
    coefficient = 219474.63137
    return coefficient * energy_in_Hartree


def convert_hartree_to_inverse_cm(energy_in_hartree: float) -> float:
    """Convert energy from Hartree (E_h)
    to wavenumber cm^{-1}
    actually a wrapper

    Args:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h

    Returns:
        energy_in_inverse_cm: the corresponding energy in
            unit of wavenumber cm^{-1}
    """
    coefficient = _convert_Hartree_to_inverse_cm(1.0)
    return coefficient * energy_in_hartree


def _convert_inverse_cm_to_hartree(energy_in_cm_inv: float) -> float:
    """Convert energy from wavenumber cm^{-1} to Hartree (E_h)

    Args:
        energy_in_inverse_cm: the corresponding energy in
            unit of wavenumber cm^{-1}

    Returns:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h
    """
    coefficient = 4.556335e-6
    return coefficient * energy_in_cm_inv


def convert_inverse_cm_to_hartree(energy_in_cm_inv: float) -> float:
    """Convert energy from wavenumber cm^{-1} to Hartree (E_h)

    Args:
        energy_in_inverse_cm: the corresponding energy in
            unit of wavenumber cm^{-1}

    Returns:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h
    """
    coefficient = _convert_inverse_cm_to_hartree(1.0)
    return coefficient * energy_in_cm_inv


def _convert_aJ_to_Hartree(energy_in_aJ: float) -> float:
    """Convert energy from aJ to Hartree (E_h)

    Args:
        energy_in_aJ: the corresponding energy in
            unit of aJ.

    Returns:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h

    """
    convert_coefficient = 0.2293710
    return energy_in_aJ * convert_coefficient


def _convert_Hartree_to_aJ(energy_in_Hartree: float) -> float:
    """Convert energy from Hartree (E_h)
    to aJ

    Args:
        energy_in_Hartree: the energy in unit of Hartree,
            E_h

    Returns:
        energy_in_aJ: the corresponding energy in
            unit of aJ.
    """
    convert_coefficient = 4.359748
    return energy_in_Hartree * convert_coefficient


def _convert_aJ_to_inverse_cm(energy_in_aJ: float) -> float:
    """Convert energy from attojoule (aJ)
    to wavenumber cm^{-1}

    Args:
        energy_in_aJ: the energy in unit of attojoule,
            aJ. (1aJ = 1e-18 J)

    Returns:
        energy_in_inverse_cm: the corresponding energy in
            unit of wavenumber cm^{-1}
    """
    coefficient = 50341.1
    energy_in_inverse_cm = coefficient * energy_in_aJ
    return energy_in_inverse_cm


def _convert_angstrom_to_a0(
    length_in_angstrom: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Convert length from angstrom (A)
    to a0, atomic unit.

    Args:
        length_in_angstrom: the length expressed in
            angstrom.

    Returns:
        length_in_a0; the length expressed under
            atomic unit a0.
    """
    # from https://www.unitconverters.net/length/angstrom-to-a-u-of-length.htm
    coefficient = 1.8897259886
    length_in_a0 = length_in_angstrom * coefficient
    return length_in_a0


def _convert_a0_to_angstrom(
    length_in_a0: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Convert length from atomic unit a0
    to angstrom (A).

    Args:
        length_in_a0; the length expressed under
            atomic unit a0.
    Returns:
        length_in_angstrom: the length expressed in
            angstrom.

    """
    # from https://www.unitconverters.net/length/a-u-of-length-to-angstrom.htm
    coefficient = 0.529177249
    length_in_angstrom = length_in_a0 * coefficient
    return length_in_angstrom

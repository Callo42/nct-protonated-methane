"""PES for CH4"""

from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch4.pes_base import IsotopomerPotentialCartesian
from neuralvib.molecule.ch4.pes_base import IsotopomerPESNormalCoor
from neuralvib.utils.convert import _convert_Hartree_to_inverse_cm


class CH4PotentialCartesian(IsotopomerPotentialCartesian):
    """CH4 PES in cartesian coordinates
    NOTE: In mass-weighted cartesian displacement coordinates.
    and all the FLATTENED cartesian displacement coordinates that appear within
    this class are ordered in the following manner:
    (C_x C_y C_z H_1x H_1y H_1z H_2x H_2y H_2z H_3x H_3y H_3z H_4x H_4y H_4z)
    with the following H atom numbering and cartesian coordinates:
    (from Mol. Phys. 37 1901(1979))

    1   ( a, a, a)
    2   (-a,-a, a)
    3   ( a,-a,-a)
    4   (-a, a, -a)
    with a = 3^{-1/2} r_e(CH)

    NOTE: ONLY the inputs to PES are in atomic unit, a0,
    while others are in angstrom, A, for consistency.
    The angles are all in radian.

    Attributes:
        self._mass_C: the mass of C in atomic unit (a.u).
        self._mass_H: the mass of H in atomic unit (a.u).
        self._pes_inter_coor: the CH4PotentialInternalCoor instanced
            for provisioning PES under symmetry internal coordinates.
        self._pes_inter_coor_func: the PES function in symmetry
            internal coordinates.
        self._r_e_CH_angstrom: the equilibrium C-H bond
            length in angstrom.
        self._S1_equilibrium: the equilibrium value of the internal
            symmetry coordinate S_1(a_1) = 1/2 * (r1 + r2 + r3 + r4),
            in angstrom.
        self._a_angstrom: the a in angstrom for locating each H atom
            used in the docstring above.
    """

    def __init__(
        self,
        select_potential: str,
    ) -> None:
        """Initialize some constants and potential

        Args:
            selected_potential: designating from which article the
                PES is used.
        """
        # NOTE: in a.u.
        self._mass_C = 12 * 1836.152673
        self._mass_H_1 = 1 * 1836.152673
        self._mass_H_2 = 1 * 1836.152673
        self._mass_H_3 = 1 * 1836.152673
        self._mass_H_4 = 1 * 1836.152673

        if select_potential == "J.Chem.Phys.102,254-261(1995)":
            # from J. Chem. Phys. 102, 254-261 (1995)
            # table I, cc-pVTZ data
            self._r_e_CH_angstrom = 1.0890
        elif select_potential == "J.Chem.Phys.102,254-261(1995)TestHarmonic":
            # from J. Chem. Phys. 102, 254-261 (1995)
            # table I, cc-pVTZ data
            # Test harmonic
            self._r_e_CH_angstrom = 1.0890
        elif select_potential == "J.Phys.Chem.A2000,104,2355-2361":
            raise NotImplementedError()

        self._S1_equilibrium = 2 * self._r_e_CH_angstrom
        self._a_angstrom = 3 ** (-1 / 2) * self._r_e_CH_angstrom

        super().__init__(
            select_potential=select_potential,
            r_e_CH_angstrom=self._r_e_CH_angstrom,
        )


class CH4PESNormalCoor(IsotopomerPESNormalCoor):
    """CH4 PES in normal coordinates.
    Ready for directly call from rectilinear normal coordinates.

    Attributes:
        self.ch4_pes_weight_cartesian: the CH4PotentialCartesian instance
        self._mass_weight_pes: the pes recieving mass weight cartesian
            displacement coordinates as input
        self._expected_eigval: (deg_of_freedom, ) the expected eigen value
            of diagnolized Hessian matrix, serves as an internal
            check of corresponding indices of normal coordinates.
        self._harmonic_omega_wavenumber: (9,) the harmonic frequencies
            omega in cm^{-1}
        self._arg_sort_index: (deg_of_freedom, ) the arg_sort result
            for reindexing Q.
            _arg_sort_index: from [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]
            to [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6].
        self._arg_sort_index_inverse: (deg_of_freedom, ) the inverse arg_sort result
            for reindexing Q.
            _arg_sort_index_inverse: from [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6].
            to [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]
        self.w_indices: the scaled omegas, omega/alpha in cm^{-1}.
        self.alpha: the omega scaling constant used for training and
            convergence chec
        self._x_e: the mass-weighted cartesian displacement coordinates
            at equilibrium configuration.
        self._U: (deg_of_freedom,deg_of_freedom) the transformation matrix
            that transforms (redundant) normal coordinates to
            mass-weighted cartesian displacement coordinates.k.

        NOTE: for reindexing Q, see docstrings of _normal_to_cartesian_displacement.

    """

    def __init__(
        self,
        select_potential: str,
        alpha: float,
    ) -> None:
        """Initialize necessary quantities.

        Args:
            selected_potential: designating from which article the
                PES is used.
            alpha: the scaling factor for training.
        """
        self.pes_weight_cartesian = CH4PotentialCartesian(
            select_potential=select_potential
        )
        self._mass_weight_pes = self.pes_weight_cartesian.pes

        if select_potential == "J.Chem.Phys.102,254-261(1995)":
            self._expected_eigval = jnp.array(
                [
                    -5.55362132e-21,
                    -3.60489726e-21,
                    -3.50818076e-21,
                    -1.75464240e-23,
                    2.09894336e-21,
                    7.98169643e-21,
                    3.74740365e-05,
                    3.74740365e-05,
                    3.74740365e-05,
                    5.12529181e-05,
                    5.12529181e-05,
                    1.91298769e-04,
                    2.06466489e-04,
                    2.06466489e-04,
                    2.06466489e-04,
                ],
                dtype=jnp.float64,
            )
            _harmonic_omega_Eh = self._expected_eigval[6::]
            self._harmonic_omega_wavenumber = _convert_Hartree_to_inverse_cm(
                jnp.sqrt(_harmonic_omega_Eh)
            )
            self._arg_sort_index = jnp.array(
                [0, 1, 2, 3, 4, 5, 12, 13, 14, 7, 8, 6, 9, 10, 11]
            )
            self._arg_sort_index_inverse = jnp.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    11,
                    9,
                    10,
                    12,
                    13,
                    14,
                    6,
                    7,
                    8,
                ]
            )
        elif select_potential == "J.Phys.Chem.A2000,104,2355-2361":
            raise NotImplementedError()

        # To make omegas from ascending order to the corresponding order
        # in paper.
        w_indices_rearrange_index = np.array([5, 3, 4, 6, 7, 8, 0, 1, 2])
        # don't divided by x_alpha here
        self.w_indices = jnp.array(
            (self._harmonic_omega_wavenumber[w_indices_rearrange_index]),
            dtype=jnp.float64,
        )
        self.alpha = alpha
        self._x_e = jnp.zeros((15,), dtype=jnp.float64)
        self._U = self._get_transfer_matrix(
            pes=self._mass_weight_pes,
            x_e=self._x_e,
            expected_eigval=self._expected_eigval,
        )

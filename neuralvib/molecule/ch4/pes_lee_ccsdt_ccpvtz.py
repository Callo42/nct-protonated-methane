"""J.Chem.Phys.102,254-261(1995) CCSD(T)-cc-pVTZ PES"""

from functools import partial
from itertools import combinations
from itertools import permutations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class InitForceConstnpArray(object):
    """Relevant to initialize np.ndarray of
    Force constants.

    Attributes:
        _convert_table: the table that could
            be used to convert subscripts in
            original paper to the array
            index used in code.

    """

    def __init__(self) -> None:
        self._convert_table = {
            "1": 0,
            "2a": 1,
            "2b": 2,
            "3x": 3,
            "3y": 4,
            "3z": 5,
            "4x": 6,
            "4y": 7,
            "4z": 8,
        }

    def _origin_subscript_to_index_list(
        self,
        origin_subscript: str,
    ) -> list:
        """convert original subscript to index_list

        Args:
            origin_subscript: the original subscripts used
                in the paper, seperated by comma.

        Returns:
            index_list: the integer list of the index
                used in code.
        """
        index_list_str = origin_subscript.split(",")
        index_list = []
        for origin_subscript in index_list_str:
            index_list.append(self._convert_table[origin_subscript])
        return index_list

    def init_quartic_F_origin(
        self,
        unique_value: dict,
        symmetry_relationship: dict,
    ) -> np.ndarray:
        """Initialize cc-pVTZ quartic force constants
        in J. Chem. Phys. 102, 254-261 (1995)
        as it is in original paper.

        NOTE: the index are rearranged as stated
        in class CH4PotentialInternalCoor.

        Args:
            unique_value: {origin_subscript: value} the
                origin unique quartic force constants
                in table II, with original subscripts
                seperated by comma as key, for example,
                {"1,2a,3z,4z":-0.04273} refers to
                F_{12a3z4z}=-0.04273.
            symmetry_relationship: {
                target_subcript: {
                    unique_subscript: coefficient,
                    unique_subscript2: coefficient2,
                    }
                }
                the quartic symmetry relationship table,
                with unique_subscript as the unique terms
                already defined by unique_value, then
                target_subscript refers to other terms that
                deduced from unique ones, with coefficient:
                target_subscript = sum_i coefficient_i * unique_subscript_i.
                E.g. {
                    "2a,2a,3y,3y":{
                        "2a,2a,3z,4z":1/4,
                        "2b,2b,3y,4y":3/4,
                    }
                }
                refers to the relationship in Mol. Phys. 60, 509(1987)
                table 4: F_{2a2a3y3y} = 1/4(F_{2a2a3z3z} + 3F_{2b2b3z3z})

        Returns:
            quartic_F_origin: (9,9,9,9) the quartic force constants
                F_ijkl in exactly the form of the original paper,
                eq(10)
        """
        quartic_F_origin = np.zeros((9, 9, 9, 9), dtype=np.float64)
        for subscript in list(unique_value.keys()):
            index_list = self._origin_subscript_to_index_list(
                origin_subscript=subscript
            )
            quartic_F_origin[
                index_list[0],
                index_list[1],
                index_list[2],
                index_list[3],
            ] = unique_value[subscript]

        for target_subscript in list(symmetry_relationship.keys()):
            target_F = 0
            relationship = symmetry_relationship[target_subscript]
            target_index_list = self._origin_subscript_to_index_list(
                origin_subscript=target_subscript
            )
            for related_subscript in list(relationship.keys()):
                coefficient = relationship[related_subscript]
                related_index_list = self._origin_subscript_to_index_list(
                    origin_subscript=related_subscript
                )
                target_F += (
                    coefficient
                    * quartic_F_origin[
                        related_index_list[0],
                        related_index_list[1],
                        related_index_list[2],
                        related_index_list[3],
                    ]
                )
            quartic_F_origin[
                target_index_list[0],
                target_index_list[1],
                target_index_list[2],
                target_index_list[3],
            ] = target_F

        return quartic_F_origin


def _jcp_102_254_1995_pVTZ_E0_cm() -> float:
    """Initialize cc-pVTZ equilibrium CCSD(T) energy
    in J. Chem. Phys. 102, 254-261 (1995)

    NOTE: Calculated directly from omegas and in wavenumber.

    Returns:
        E_0_cm: E_0 in wavenumber.
    """
    E_0_cm = 9835.0
    return E_0_cm


def _jcp_102_254_1995_pVTZ_E0() -> float:
    """Initialize cc-pVTZ equilibrium CCSD(T) energy
    in J. Chem. Phys. 102, 254-261 (1995)

    NOTE: the original E0 in paper is listed
    under Table I. and is in Hartree unit.

    Returns:
        E_0_aJ: E_0 in Hartree.
    """
    E_0_Hartree = -40.438099
    return E_0_Hartree


def _jcp_102_254_1995_pVTZ_harmonic() -> np.ndarray:
    """Initialize cc-pVTZ harmonic frequencis
    in J. Chem. Phys. 102, 254-261 (1995)
    in wavenumber

    NOTE: the index are rearranged as stated
    in class CH4PotentialInternalCoor.

    Returns:
        omega: (9,) the harmonic frequencis corresponding to
            each symmetry internal coordinates.
    """
    omega = np.array(
        [
            3034.7,
            1570.8,
            1570.8,
            3153.9,
            3153.9,
            3153.9,
            1344.0,
            1344.0,
            1344.0,
        ],
        dtype=np.float64,
    )
    return omega


def _jcp_102_254_1995_pVTZ_quadratic() -> np.ndarray:
    """Initialize cc-pVTZ quadratic force constants
    in J. Chem. Phys. 102, 254-261 (1995)
    NOTE: the index are rearranged as stated
    in class CH4PotentialInternalCoor.

    NOTE: the function firstly define force constants
    F as it is in the origin paper, THEN transformed
    into F_new to fit the potential expansion that is
    actually used in calculation. See comments in
    class CH4PotentialInternalCoor for details.

    NOTE: all the force constants are
    in the unit of aJ.

    Returns:
        quadratic_F_new: (9,9) the quadratic force constants
            expressed in new summation.
    """
    quadratic_F_origin = np.zeros((9, 9), dtype=np.float64)
    quadratic_F_new = np.zeros((9, 9), dtype=np.float64)

    # ref Mol. Phys. 37, 1901(1979), eq(5)
    # F11
    quadratic_F_origin[0, 0] = 5.46865
    # F22
    quadratic_F_origin[1, 1] = 0.57919
    quadratic_F_origin[2, 2] = 0.57919
    # F33
    quadratic_F_origin[3, 3] = 5.36602
    quadratic_F_origin[4, 4] = 5.36602
    quadratic_F_origin[5, 5] = 5.36602
    # F34
    quadratic_F_origin[3, 6] = -0.21036
    quadratic_F_origin[4, 7] = -0.21036
    quadratic_F_origin[5, 8] = -0.21036
    # F44
    quadratic_F_origin[6, 6] = 0.53227
    quadratic_F_origin[7, 7] = 0.53227
    quadratic_F_origin[8, 8] = 0.53227

    # convert to fit the summation under
    # restriction i<=j
    for i in range(9):
        for j in range(9):
            indices = (i, j)
            weight = len(set(permutations(indices))) / 2
            quadratic_F_new[indices] = weight * quadratic_F_origin[indices]

    return quadratic_F_new


def _jcp_102_254_1995_pVTZ_cubic() -> np.ndarray:
    """Initialize cc-pVTZ cubic force constants
    in J. Chem. Phys. 102, 254-261 (1995)
    NOTE: the index are rearranged as stated
    in class CH4PotentialInternalCoor.

    NOTE: the function firstly define force constants
    F as it is in the origin paper, THEN transformed
    into F_new to fit the potential expansion that is
    actually used in calculation. See comments in
    class CH4PotentialInternalCoor for details.

    NOTE: all the force constants are
    in the unit of aJ.

    Returns:
        cubic_F_new: (9,9,9) the cubic force constants
            expressed in new summation.
    """
    cubic_F_origin = np.zeros((9, 9, 9), dtype=np.float64)
    cubic_F_new = np.zeros((9, 9, 9), dtype=np.float64)

    # unique cubic force constants
    # following the order in table II.
    cubic_F_origin[0, 0, 0] = -15.17114
    cubic_F_origin[0, 1, 1] = -0.25438
    cubic_F_origin[0, 3, 3] = -15.49772
    cubic_F_origin[0, 3, 6] = 0.06598
    cubic_F_origin[0, 6, 6] = -0.22556
    cubic_F_origin[1, 1, 1] = 0.09116
    cubic_F_origin[1, 5, 5] = -0.35605
    cubic_F_origin[1, 5, 8] = 0.18004
    cubic_F_origin[1, 8, 8] = -0.34330
    cubic_F_origin[3, 4, 5] = -15.57546
    cubic_F_origin[3, 4, 8] = -0.21811
    cubic_F_origin[3, 7, 8] = -0.09616
    cubic_F_origin[6, 7, 8] = 0.34391

    # others with relationships
    # Mol. Phys. 37, 1901 (1979) table 2
    # F122
    cubic_F_origin[0, 2, 2] = cubic_F_origin[0, 1, 1]
    # F133
    cubic_F_origin[0, 4, 4] = cubic_F_origin[0, 3, 3]
    cubic_F_origin[0, 5, 5] = cubic_F_origin[0, 3, 3]
    # F134
    cubic_F_origin[0, 4, 7] = cubic_F_origin[0, 3, 6]
    cubic_F_origin[0, 5, 8] = cubic_F_origin[0, 3, 6]
    # F144
    cubic_F_origin[0, 7, 7] = cubic_F_origin[0, 6, 6]
    cubic_F_origin[0, 8, 8] = cubic_F_origin[0, 6, 6]
    # F222
    cubic_F_origin[1, 2, 2] = -1 * cubic_F_origin[1, 1, 1]
    # F233
    cubic_F_origin[1, 3, 3] = -(1 / 2) * cubic_F_origin[1, 5, 5]
    cubic_F_origin[1, 4, 4] = -(1 / 2) * cubic_F_origin[1, 5, 5]
    cubic_F_origin[2, 3, 3] = (np.sqrt(3) / 2) * cubic_F_origin[1, 5, 5]
    cubic_F_origin[2, 4, 4] = -(np.sqrt(3) / 2) * cubic_F_origin[1, 5, 5]
    # F234
    cubic_F_origin[1, 3, 6] = -(1 / 2) * cubic_F_origin[1, 5, 8]
    cubic_F_origin[1, 4, 7] = -(1 / 2) * cubic_F_origin[1, 5, 8]
    cubic_F_origin[2, 3, 6] = (np.sqrt(3) / 2) * cubic_F_origin[1, 5, 8]
    cubic_F_origin[2, 4, 7] = -(np.sqrt(3) / 2) * cubic_F_origin[1, 5, 8]
    # F244
    cubic_F_origin[1, 6, 6] = -(1 / 2) * cubic_F_origin[1, 8, 8]
    cubic_F_origin[1, 7, 7] = -(1 / 2) * cubic_F_origin[1, 8, 8]
    cubic_F_origin[2, 6, 6] = (np.sqrt(3) / 2) * cubic_F_origin[1, 8, 8]
    cubic_F_origin[2, 7, 7] = -(np.sqrt(3) / 2) * cubic_F_origin[1, 8, 8]
    # F334
    cubic_F_origin[4, 5, 6] = cubic_F_origin[3, 4, 8]
    cubic_F_origin[3, 5, 7] = cubic_F_origin[3, 4, 8]
    # F344
    cubic_F_origin[4, 6, 8] = cubic_F_origin[3, 7, 8]
    cubic_F_origin[5, 6, 7] = cubic_F_origin[3, 7, 8]

    # convert to fit the summation under
    # restriction i<=j<=k
    for i in range(9):
        for j in range(9):
            for k in range(9):
                indices = (i, j, k)
                weight = len(set(permutations(indices))) / 6
                cubic_F_new[indices] = weight * cubic_F_origin[indices]
    return cubic_F_new


def _jcp_102_254_1995_pVTZ_quartic() -> np.ndarray:
    """Initialize cc-pVTZ quartic force constants
    in J. Chem. Phys. 102, 254-261 (1995)
    NOTE: the index are rearranged as stated
    in class CH4PotentialInternalCoor.

    NOTE: the function firstly define force constants
    F as it is in the origin paper, THEN transformed
    into F_new to fit the potential expansion that is
    actually used in calculation. See comments in
    class CH4PotentialInternalCoor for details.

    NOTE: all the force constants are
    in the unit of aJ.

    Returns:
        quartic_F_new: (9,9,9,9) the quartic force constants
            expressed in new summation.
    """
    init_F = InitForceConstnpArray()
    quartic_F_new = np.zeros((9, 9, 9, 9), dtype=np.float64)
    unique_value = {
        "1,1,1,1": 37.41710,
        "1,1,2a,2a": -0.01264,
        "1,1,3x,3x": 39.80537,
        "1,1,3x,4x": 0.21886,
        "1,1,4x,4x": 0.05929,
        "1,2a,2a,2a": -0.06147,
        "1,2a,3z,3z": 0.06888,
        "1,2a,3z,4z": -0.04273,
        "1,2a,4z,4z": 0.14474,
        "1,3x,3y,3z": 40.64855,
        "1,3x,3y,4z": 0.11859,
        "1,3x,4y,4z": -0.02093,
        "1,4x,4y,4z": -0.16567,
        "2a,2a,2a,2a": 0.17308,
        "2a,2a,3z,3z": -0.19706,
        "2b,2b,3z,3z": -0.43249,
        "2a,2a,3z,4z": -0.13777,
        "2b,2b,3z,4z": -0.04883,
        "2a,2a,4z,4z": 0.01744,
        "2b,2b,4z,4z": 0.37668,
        "2a,3x,3y,4z": 0.27554,
        "2a,3z,4x,4y": -0.06801,
        "3x,3x,3x,3x": 41.04703,
        "3x,3x,3y,3y": 41.14033,
        "3x,3x,3x,4x": 0.18485,
        "3x,3x,3y,4y": 0.32611,
        "3x,3x,4x,4x": 0.09940,
        "3x,3y,4x,4y": -0.00523,
        "3x,3x,4y,4y": -0.24415,
        "3x,4x,4x,4x": -0.39587,
        "3x,4x,4y,4y": -0.14823,
        "4x,4x,4x,4x": 0.49876,
        "4x,4x,4y,4y": 0.70977,
    }
    # NOTE: the symmetry relationships of F2334
    # in paper Mol. Phys. 60, 509(1987) has
    # sing ERROR! this particular relationship
    # is taken from J. Chem. Phys. 102, 254-261 (1995)
    symmetry_relationship = {
        "1,1,2b,2b": {
            "1,1,2a,2a": 1,
        },
        "1,1,3y,3y": {
            "1,1,3x,3x": 1,
        },
        "1,1,3z,3z": {
            "1,1,3x,3x": 1,
        },
        "1,1,3y,4y": {
            "1,1,3x,4x": 1,
        },
        "1,1,3z,4z": {
            "1,1,3x,4x": 1,
        },
        "1,1,4y,4y": {
            "1,1,4x,4x": 1,
        },
        "1,1,4z,4z": {
            "1,1,4x,4x": 1,
        },
        "1,2a,2b,2b": {
            "1,2a,2a,2a": -1,
        },
        "1,2a,3x,3x": {
            "1,2a,3z,3z": -1 / 2,
        },
        "1,2a,3y,3y": {
            "1,2a,3z,3z": -1 / 2,
        },
        "1,2b,3x,3x": {
            "1,2a,3z,3z": np.sqrt(3) / 2,
        },
        "1,2b,3y,3y": {
            "1,2a,3z,3z": -np.sqrt(3) / 2,
        },
        "1,2a,3x,4x": {
            "1,2a,3z,4z": -1 / 2,
        },
        "1,2a,3y,4y": {
            "1,2a,3z,4z": -1 / 2,
        },
        "1,2b,3x,4x": {
            "1,2a,3z,4z": np.sqrt(3) / 2,
        },
        "1,2b,3y,4y": {
            "1,2a,3z,4z": -np.sqrt(3) / 2,
        },
        "1,2a,4x,4x": {
            "1,2a,4z,4z": -1 / 2,
        },
        "1,2a,4y,4y": {
            "1,2a,4z,4z": -1 / 2,
        },
        "1,2b,4x,4x": {
            "1,2a,4z,4z": np.sqrt(3) / 2,
        },
        "1,2b,4y,4y": {
            "1,2a,4z,4z": -np.sqrt(3) / 2,
        },
        "1,3x,3z,4y": {
            "1,3x,3y,4z": 1,
        },
        "1,3y,3z,4x": {
            "1,3x,3y,4z": 1,
        },
        "1,3y,4x,4z": {
            "1,3x,4y,4z": 1,
        },
        "1,3z,4x,4y": {
            "1,3x,4y,4z": 1,
        },
        "2b,2b,2b,2b": {
            "2a,2a,2a,2a": 1,
        },
        "2a,2a,2b,2b": {
            "2a,2a,2a,2a": 1 / 3,
        },
        "2a,2a,3y,3y": {
            "2a,2a,3z,3z": 1 / 4,
            "2b,2b,3z,3z": 3 / 4,
        },
        "2a,2a,3x,3x": {
            "2a,2a,3z,3z": 1 / 4,
            "2b,2b,3z,3z": 3 / 4,
        },
        "2b,2b,3y,3y": {
            "2a,2a,3z,3z": 3 / 4,
            "2b,2b,3z,3z": 1 / 4,
        },
        "2b,2b,3x,3x": {
            "2a,2a,3z,3z": 3 / 4,
            "2b,2b,3z,3z": 1 / 4,
        },
        "2a,2b,3y,3y": {
            "2a,2a,3z,3z": np.sqrt(3) / 4,
            "2b,2b,3z,3z": -np.sqrt(3) / 4,
        },
        "2a,2b,3x,3x": {
            "2a,2a,3z,3z": -np.sqrt(3) / 4,
            "2b,2b,3z,3z": np.sqrt(3) / 4,
        },
        "2a,2a,3y,4y": {
            "2a,2a,3z,4z": 1 / 4,
            "2b,2b,3z,4z": 3 / 4,
        },
        "2a,2a,3x,4x": {
            "2a,2a,3z,4z": 1 / 4,
            "2b,2b,3z,4z": 3 / 4,
        },
        "2b,2b,3y,4y": {
            "2a,2a,3z,4z": 3 / 4,
            "2b,2b,3z,4z": 1 / 4,
        },
        "2b,2b,3x,4x": {
            "2a,2a,3z,4z": 3 / 4,
            "2b,2b,3z,4z": 1 / 4,
        },
        "2a,2b,3y,4y": {
            "2a,2a,3z,4z": np.sqrt(3) / 4,
            "2b,2b,3z,4z": -np.sqrt(3) / 4,
        },
        "2a,2b,3x,4x": {
            "2a,2a,3z,4z": -np.sqrt(3) / 4,
            "2b,2b,3z,4z": np.sqrt(3) / 4,
        },
        "2a,2a,4y,4y": {
            "2a,2a,4z,4z": 1 / 4,
            "2b,2b,4z,4z": 3 / 4,
        },
        "2a,2a,4x,4x": {
            "2a,2a,4z,4z": 1 / 4,
            "2b,2b,4z,4z": 3 / 4,
        },
        "2b,2b,4y,4y": {
            "2a,2a,4z,4z": 3 / 4,
            "2b,2b,4z,4z": 1 / 4,
        },
        "2b,2b,4x,4x": {
            "2a,2a,4z,4z": 3 / 4,
            "2b,2b,4z,4z": 1 / 4,
        },
        "2a,2b,4y,4y": {
            "2a,2a,4z,4z": np.sqrt(3) / 4,
            "2b,2b,4z,4z": -np.sqrt(3) / 4,
        },
        "2a,2b,4x,4x": {
            "2a,2a,4z,4z": -np.sqrt(3) / 4,
            "2b,2b,4z,4z": np.sqrt(3) / 4,
        },
        "2a,3y,3z,4x": {
            "2a,3x,3y,4z": -1 / 2,
        },
        "2a,3x,3z,4y": {
            "2a,3x,3y,4z": -1 / 2,
        },
        "2b,3y,3z,4x": {
            "2a,3x,3y,4z": np.sqrt(3) / 2,
        },
        "2b,3x,3z,4y": {
            "2a,3x,3y,4z": -np.sqrt(3) / 2,
        },
        "2a,3x,4y,4z": {
            "2a,3z,4x,4y": -1 / 2,
        },
        "2a,3y,4x,4z": {
            "2a,3z,4x,4y": -1 / 2,
        },
        "2b,3x,4y,4z": {
            "2a,3z,4x,4y": np.sqrt(3) / 2,
        },
        "2b,3y,4x,4z": {
            "2a,3z,4x,4y": -np.sqrt(3) / 2,
        },
        "3y,3y,3y,3y": {
            "3x,3x,3x,3x": 1,
        },
        "3z,3z,3z,3z": {
            "3x,3x,3x,3x": 1,
        },
        "3x,3x,3z,3z": {
            "3x,3x,3y,3y": 1,
        },
        "3y,3y,3z,3z": {
            "3x,3x,3y,3y": 1,
        },
        "3y,3y,3y,4y": {
            "3x,3x,3x,4x": 1,
        },
        "3z,3z,3z,4z": {
            "3x,3x,3x,4x": 1,
        },
        "3y,3y,3z,4z": {
            "3x,3x,3y,4y": 1,
        },
        "3x,3y,3y,4x": {
            "3x,3x,3y,4y": 1,
        },
        "3x,3z,3z,4x": {
            "3x,3x,3y,4y": 1,
        },
        "3y,3z,3z,4y": {
            "3x,3x,3y,4y": 1,
        },
        "3x,3x,3z,4z": {
            "3x,3x,3y,4y": 1,
        },
        "3y,3y,4y,4y": {
            "3x,3x,4x,4x": 1,
        },
        "3z,3z,4z,4z": {
            "3x,3x,4x,4x": 1,
        },
        "3x,3z,4x,4z": {
            "3x,3y,4x,4y": 1,
        },
        "3y,3z,4y,4z": {
            "3x,3y,4x,4y": 1,
        },
        "3y,3y,4x,4x": {
            "3x,3x,4y,4y": 1,
        },
        "3z,3z,4y,4y": {
            "3x,3x,4y,4y": 1,
        },
        "3y,3y,4z,4z": {
            "3x,3x,4y,4y": 1,
        },
        "3x,3x,4z,4z": {
            "3x,3x,4y,4y": 1,
        },
        "3z,3z,4x,4x": {
            "3x,3x,4y,4y": 1,
        },
        "3y,4y,4y,4y": {
            "3x,4x,4x,4x": 1,
        },
        "3z,4z,4z,4z": {
            "3x,4x,4x,4x": 1,
        },
        "3x,4x,4z,4z": {
            "3x,4x,4y,4y": 1,
        },
        "3y,4x,4x,4y": {
            "3x,4x,4y,4y": 1,
        },
        "3y,4y,4z,4z": {
            "3x,4x,4y,4y": 1,
        },
        "3z,4x,4x,4z": {
            "3x,4x,4y,4y": 1,
        },
        "3z,4y,4y,4z": {
            "3x,4x,4y,4y": 1,
        },
        "4y,4y,4y,4y": {
            "4x,4x,4x,4x": 1,
        },
        "4z,4z,4z,4z": {
            "4x,4x,4x,4x": 1,
        },
        "4x,4x,4z,4z": {
            "4x,4x,4y,4y": 1,
        },
        "4y,4y,4z,4z": {
            "4x,4x,4y,4y": 1,
        },
    }

    quartic_F_origin = init_F.init_quartic_F_origin(
        unique_value=unique_value,
        symmetry_relationship=symmetry_relationship,
    )

    # convert to fit the summation under
    # restriction i<=j<=k<=l
    for i in range(9):
        for j in range(9):
            for k in range(9):
                for last_index in range(9):
                    indices = (i, j, k, last_index)
                    weight = len(set(permutations(indices))) / 24
                    quartic_F_new[indices] = weight * quartic_F_origin[indices]
    return quartic_F_new


class PotentialInternalCoor(object):
    """Supporting CH4 potential energy in symmetry
    internal coordinates as presented in
    J. Chem. Phys. 102, 254-261 (1995)

    NOTE: Here the sub index of symmetry internal
    coordinates, S_{i}, from eq(1) to eq(9) in the paper are reassigned
    as follow for convenience:
    || subscript in origin paper || array index in code ||
                ||          1    ||  0           ||
                ||          2a   ||  1           ||
                ||          2b   ||  2           ||
                ||          3x   ||  3           ||
                ||          3y   ||  4           ||
                ||          3z   ||  5           ||
                ||          4x   ||  6           ||
                ||          4y   ||  7           ||
                ||          4z   ||  8           ||
    For example, for the array sym_inter_coor with shape (9,)
    the relationship between it and original symmetry internal
    coordinates would be retrieved like
        S_{3y} == sym_inter_coor[4]

    NOTE: The potential expansion, eq(10) of the origin paper:
    V = E_0 + 1/2 sum_{ij} F_{ij} DeltaSi DeltaSj
            + 1/6 sum_{ijk} F_{ijk} DeltaSi DeltaSj DeltaSk
            + 1/24 sum_{ijkl} F_{ijkl} DeltaSi DeltaSj DeltaSk DeltaSl
    is converted to the following form, with restrictions to only
    sum the terms with no replication:

    V = E_0 + sum_{i<=j} F_new{ij} DeltaSi DeltaSj
            + sum_{i<=j<=k} F_new{ijk} DeltaSi DeltaSj DeltaSk
            + sum_{i<=j<=k<=l} F_new{ijkl} DeltaSi DeltaSj DeltaSk DeltaSl
    where F_new are the new force constants after conversion.

    NOTE: all the force constants are
    in the unit of aJ.

    Attributes:
        self._E0: the E_0 in PES
        self._harmonic_omega: (9,) the harmonic frequencies
            omega
        self._quadratic_F_new: (9,9) the quadratic force
            constants, in the constraints of new PES
        self._cubic_F_new: (9,9,9) the cubic force
            constants, in the constraints of new PES
        self._quartic_F_new: (9,9,9,9) the quartic force
            constants, in the constraints of new PES


    """

    def __init__(
        self,
        potential: str,
    ) -> None:
        """Initialize the potential
        from J. Chem. Phys. 102, 254-261 (1995)

        Args:
            potential: designating from which article the
                PES is used.
        """
        _potentials_available = [
            "J.Chem.Phys.102,254-261(1995)",
            "J.Chem.Phys.102,254-261(1995)TestHarmonic",
            "J.Phys.Chem.A2000,104,2355-2361",
        ]
        self._potentials_available = _potentials_available
        if potential not in _potentials_available:
            raise ValueError(
                f"Desinated potential: {potential} not found!"
                f"\n Currently avialable: {_potentials_available}"
            )
        elif potential == "J.Chem.Phys.102,254-261(1995)":
            self._harmonic_omega = _jcp_102_254_1995_pVTZ_harmonic()
            self._quadratic_F_new = _jcp_102_254_1995_pVTZ_quadratic()
            self._cubic_F_new = _jcp_102_254_1995_pVTZ_cubic()
            self._quartic_F_new = _jcp_102_254_1995_pVTZ_quartic()
        elif potential == "J.Chem.Phys.102,254-261(1995)TestHarmonic":
            self._harmonic_omega = _jcp_102_254_1995_pVTZ_harmonic()
            self._quadratic_F_new = _jcp_102_254_1995_pVTZ_quadratic()
            self._cubic_F_new = jnp.zeros((9, 9, 9), dtype=jnp.float64)
            self._quartic_F_new = jnp.zeros((9, 9, 9, 9), dtype=jnp.float64)
        elif potential == "J.Phys.Chem.A2000,104,2355-2361":
            raise NotImplementedError()

    def pes(
        self,
        delta_si: jax.Array,
    ) -> float:
        """the target PES
        NOTE: energy unit in aJ.

        Args:
            delta_si: (9,) the displacement in
                symmetry internal coordinates, with
                unit A and rad.

        Returns:
            potential_energy: corresponding potential
                energy in aJ.
        """
        einsum1 = jnp.einsum("ij,i,j", self._quadratic_F_new, delta_si, delta_si)
        einsum2 = jnp.einsum(
            "ijk,i,j,k", self._cubic_F_new, delta_si, delta_si, delta_si
        )
        einsum3 = jnp.einsum(
            "ijkl,i,j,k,l", self._quartic_F_new, delta_si, delta_si, delta_si, delta_si
        )
        # print(
        #         f"delta_si = {delta_si}"
        #         f"einsum1 = {einsum1}"
        #         f"einsum2 = {einsum2}"
        #         f"einsum3 = {einsum3}"
        # )
        # potential_energy = (self._E0 + einsum1 + einsum2 + einsum3)
        potential_energy = einsum1 + einsum2 + einsum3
        return potential_energy

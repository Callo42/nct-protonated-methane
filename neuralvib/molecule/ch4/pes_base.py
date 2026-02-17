"""PES for CH4"""

from itertools import combinations
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch4.pes_lee_ccsdt_ccpvtz import PotentialInternalCoor
from neuralvib.molecule.ch4.configuration import _calculate_angle
from neuralvib.utils.convert import _convert_a0_to_angstrom
from neuralvib.utils.convert import _convert_angstrom_to_a0
from neuralvib.utils.convert import _convert_aJ_to_Hartree
from neuralvib.utils.convert import _convert_Hartree_to_inverse_cm
from neuralvib.molecule.ch4.trans2normal import _normal_to_cartesian_displacement


class IsotopomerPotentialCartesian(object):
    """CH4 and isotopomers PES in cartesian coordinates
    from J. Chem. Phys. 102, 254-261 (1995)
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
        self._alpha_e_rad: the equilibrium bond angle
            alpha_e in rad.
        self._pes_inter_coor: the CH4PotentialInternalCoor instanced
            for provisioning PES under symmetry internal coordinates.
        self._pes_inter_coor_func: the PES function in symmetry
            internal coordinates.
        self._equilibirium_config: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, flattened, in angstrom.

    """

    def __init__(
        self,
        select_potential: str,
        r_e_CH_angstrom: float,
    ) -> None:
        """Initialize some constants and potential

        Args:
            selected_potential: designating from which article the
                PES is used.
            r_e_CH_angstrom: the equilibrium C-H bond
                length in angstrom.
        """
        self._alpha_e_rad = np.deg2rad(109.5)
        self._pes_inter_coor = PotentialInternalCoor(potential=select_potential)
        self._pes_inter_coor_func = self._pes_inter_coor.pes
        self._equilibrium_config = jnp.array(
            [
                0.0,
                0.0,
                0.0,
                self._a_angstrom,
                self._a_angstrom,
                self._a_angstrom,
                -self._a_angstrom,
                -self._a_angstrom,
                self._a_angstrom,
                self._a_angstrom,
                -self._a_angstrom,
                -self._a_angstrom,
                -self._a_angstrom,
                self._a_angstrom,
                -self._a_angstrom,
            ],
            dtype=jnp.float64,
        )

    def _get_mass_weight(
        self,
        direction: str,
    ) -> np.ndarray:
        """Generate mass weight for mass-weighted cartesian
        displacement coordinates.

        Args:
            direction: denote which direction to perform conversion
                if "weight2config" then config_coor = coefficient * mass_weighted_coor
                if "config2weight" then mass_weighted_coor = coefficient * config_coor

        Returns:
            coefficient: (15,) in the same order as flattened
                configuration cartesian displacement coordinates.
        """
        mass_C = self._mass_C
        mass_H_1 = self._mass_H_1
        mass_H_2 = self._mass_H_2
        mass_H_3 = self._mass_H_3
        mass_H_4 = self._mass_H_4
        if direction == "weight2config":
            coefficient = 1.0 / np.sqrt(
                np.array(
                    [
                        mass_C,
                        mass_C,
                        mass_C,
                        mass_H_1,
                        mass_H_1,
                        mass_H_1,
                        mass_H_2,
                        mass_H_2,
                        mass_H_2,
                        mass_H_3,
                        mass_H_3,
                        mass_H_3,
                        mass_H_4,
                        mass_H_4,
                        mass_H_4,
                    ],
                    dtype=np.float64,
                )
            )
        elif direction == "config2weight":
            coefficient = np.sqrt(
                np.array(
                    [
                        mass_C,
                        mass_C,
                        mass_C,
                        mass_H_1,
                        mass_H_1,
                        mass_H_1,
                        mass_H_2,
                        mass_H_2,
                        mass_H_2,
                        mass_H_3,
                        mass_H_3,
                        mass_H_3,
                        mass_H_4,
                        mass_H_4,
                        mass_H_4,
                    ],
                    dtype=np.float64,
                )
            )
        else:
            raise ValueError(f"Unexpected direction={direction}")
        return coefficient

    def _ccd2mwcd(
        self,
        configuration_cartesian_displacement: jax.Array,
    ) -> jax.Array:
        """configuration cartesian displacement to mass-weight cartesian displacement

        Args:
            configuration_cartesian_displacement: (15,) the flattened
                configureation cartesian displacement coordinates of CH4 with
                corresponding relationship stated in the docstring
                of IsotopomerPotentialCartesian.

        Returns:
            mass_weight_cartesian_displacement: (15,) the flattened mass
                weight cartesian displacement coordinates of CH4, with
                corresponding relationship stated in the docstring
                of IsotopomerPotentialCartesian.
        """
        coefficient = self._get_mass_weight(direction="config2weight")
        mass_weight_cartesian_displacement = jnp.multiply(
            coefficient, configuration_cartesian_displacement
        )
        return mass_weight_cartesian_displacement

    def _mwcd2ccd(
        self,
        mass_weight_cartesian_displacement: jax.Array,
    ) -> jax.Array:
        """mass-weight cartesian displacement to configuration cartesian displacement

        Args:
            mass_weight_cartesian_displacement: (15,) the flattened mass
                weight cartesian displacement coordinates of CH4, with
                corresponding relationship stated in the docstring
                of IsotopomerPotentialCartesian.

        Returns:
            configuration_cartesian_displacement: (15,) the flattened
                configureation cartesian displacement coordinates of CH4 with
                corresponding relationship stated in the docstring
                of IsotopomerPotentialCartesian.
        """
        # retrive configuration displacement
        coefficient = self._get_mass_weight(direction="weight2config")
        configuration_cartesian_displacement = jnp.multiply(
            coefficient, mass_weight_cartesian_displacement
        )
        return configuration_cartesian_displacement

    def _ccd2sid(
        self,
        configuration_cartesian_displacement: jax.Array,
    ) -> jax.Array:
        """configuration cartesian displacement to symmetry internal displacement

        Args:
            configuration_cartesian_displacement: (15,) the flattened
                configureation cartesian displacement coordinates of CH4 with
                corresponding relationship stated in the docstring
                of IsotopomerPotentialCartesian.

        Returns:
            symmetry_internal_displacement: (9,) the symmetry internal displacement
                coordinates w.r.t. eq(1) to eq(9) in the paper.
        """
        configuration_cartesian = (
            self._equilibrium_config + configuration_cartesian_displacement
        )
        configuration_reshaped = configuration_cartesian.reshape(5, 3)
        coor_C = configuration_reshaped[0]
        coor_H_1 = configuration_reshaped[1]
        coor_H_2 = configuration_reshaped[2]
        coor_H_3 = configuration_reshaped[3]
        coor_H_4 = configuration_reshaped[4]

        r1_vector = coor_H_1 - coor_C
        r2_vector = coor_H_2 - coor_C
        r3_vector = coor_H_3 - coor_C
        r4_vector = coor_H_4 - coor_C
        # same notations as in paper.
        r1 = jnp.linalg.norm(r1_vector)
        r2 = jnp.linalg.norm(r2_vector)
        r3 = jnp.linalg.norm(r3_vector)
        r4 = jnp.linalg.norm(r4_vector)

        alphas = []
        for vec_1, vec_2 in combinations(
            [r1_vector, r2_vector, r3_vector, r4_vector],
            r=2,
        ):
            angle = _calculate_angle(vec_1=vec_1, vec_2=vec_2)
            alphas.append(angle)
        [alpha_12, alpha_13, alpha_14, alpha_23, alpha_24, alpha_34] = alphas

        symmetry_internal_current = jnp.array(
            [
                (r1 + r2 + r3 + r4) / 2,
                (
                    2 * alpha_12
                    - alpha_13
                    - alpha_14
                    - alpha_23
                    - alpha_24
                    + 2 * alpha_34
                )
                / jnp.sqrt(12),
                (alpha_13 - alpha_14 - alpha_23 + alpha_24) / 2,
                (r1 - r2 + r3 - r4) / 2,
                (r1 - r2 - r3 + r4) / 2,
                (r1 + r2 - r3 - r4) / 2,
                (alpha_24 - alpha_13) / jnp.sqrt(2),
                (alpha_23 - alpha_14) / jnp.sqrt(2),
                (alpha_34 - alpha_12) / jnp.sqrt(2),
            ],
            dtype=jnp.float64,
        )
        symmetry_internal_eq = jnp.array(
            [
                self._S1_equilibrium,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=jnp.float64,
        )
        symmetry_internal_displacement = (
            symmetry_internal_current - symmetry_internal_eq
        )
        return symmetry_internal_displacement

    def pes(
        self,
        mass_weight_cartesian_displacement_a0: jax.Array,
    ) -> float:
        """the PES that called with mass weight cartesian displacements
        coordinates (flattened) as input.

        Args:
            mass_weight_cartesian_displacement_a0: (15,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the class.
                NOTE: in atomic unit!

        Returns:
            potential_energy: corresponding potential energy in Hartree.
        """
        massweight_cartesiandisplacement_angstrom = _convert_a0_to_angstrom(
            length_in_a0=mass_weight_cartesian_displacement_a0
        )
        configuration_cartesian_displacement = self._mwcd2ccd(
            mass_weight_cartesian_displacement=massweight_cartesiandisplacement_angstrom
        )
        symmetry_internal_displacement = self._ccd2sid(
            configuration_cartesian_displacement=configuration_cartesian_displacement
        )
        potential_energy = self._pes_inter_coor_func(
            delta_si=symmetry_internal_displacement
        )
        potential_energy_Hartree = _convert_aJ_to_Hartree(potential_energy)
        return potential_energy_Hartree


class IsotopomerPESNormalCoor(object):
    """CH4 and isotopomer PES in normal coordinates.
    Ready for directly call from rectilinear normal coordinates.

    Attributes:
        self._x_e: the mass-weighted cartesian displacement coordinates
            at equilibrium configuration.
        self._U: (deg_of_freedom,deg_of_freedom) the transformation matrix
            that transforms (redundant) normal coordinates to
            mass-weighted cartesian displacement coordinates.

        NOTE: for reindexing Q, see docstrings of _normal_to_cartesian_displacement.

    """

    def __init__(
        self,
    ) -> None:
        """Initialize necessary quantities."""

    def _get_transfer_matrix(
        self,
        pes: Callable,
        x_e: jax.Array,
        expected_eigval: jax.Array,
    ) -> jax.Array:
        """Same as _normal_to_cartesian_displacement.

        Args:
            pes: the function that take x as input, returns potential
                energy.
            x_e: (deg_of_freedom, ) the x at equilibrium configuration.
            expected_eigval: (deg_of_freedom, ) the expected eigen value
                of diagnolized Hessian matrix, serves as an internal
                check of corresponding indices of normal coordinates.

        Returns:
            U: (deg_of_freedom,deg_of_freedom) the transformation matrix
                that transforms (redundant) normal coordinates to
                mass-weighted cartesian displacement coordinates
        """
        return _normal_to_cartesian_displacement(
            pes=pes,
            x_e=x_e,
            expected_eigval=expected_eigval,
        )

    def _convert_scaled_x_to_normal(
        self,
        scaled_x: jax.Array,
    ) -> jax.Array:
        """Convert scaled x to normal coordinates

        Args:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.

        Returns:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        beta = _convert_Hartree_to_inverse_cm(1)
        scaled_x_flattened = jnp.hstack(scaled_x)
        normal_coor = np.sqrt(beta) * scaled_x_flattened / np.sqrt(self.alpha)
        return normal_coor

    def _convert_scaled_normal_to_normal(
        self,
        scaled_normal: jax.Array,
    ) -> jax.Array:
        """Convert scaled normal coordinates to normal coordinates

        Args:
            scaled_normal: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.

        Returns:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        scaled_normal_flattened = jnp.hstack(scaled_normal)
        normal_coor = scaled_normal_flattened / np.sqrt(self.alpha)
        return normal_coor

    def _convert_normal_to_scaled_normal(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to scaled normal coordinates

        Args:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            scaled_normal: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.
        """
        scaled_x_flattened = np.sqrt(self.alpha) * normal_coor
        scaled_normal = scaled_x_flattened.reshape(9, 1)
        return scaled_normal

    def _convert_normal_to_scaled_x(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to scaled x

        Args:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.
        """
        beta = _convert_Hartree_to_inverse_cm(1)
        scaled_x_flattened = np.sqrt(self.alpha) * normal_coor / np.sqrt(beta)
        scaled_x = scaled_x_flattened.reshape(9, 1)
        return scaled_x

    def _convert_normal_to_mass_weight(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to
        mass weighted cartesian displacement coordinates.

        Args:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            mass_weight_cartesian_displacement_a0: (15,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the class.
                NOTE: in atomic unit!
        """
        zeros = jnp.zeros(6)
        Q_redundant = jnp.concatenate((zeros, normal_coor))
        Q_reindexed_redundant = Q_redundant[self._arg_sort_index]
        mass_weight_cartesian_displacement_a0 = jnp.einsum(
            "ij,j", self._U, Q_reindexed_redundant
        )
        return mass_weight_cartesian_displacement_a0

    def _convert_mass_weight_to_normal(
        self,
        mass_weight_cartesian_displacement_a0: jax.Array,
    ) -> jax.Array:
        """Convert mass weighted cartesian displacement coordinates to
        normal coordinates.

        Args:
            mass_weight_cartesian_displacement_a0: (15,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the class.
                NOTE: in atomic unit!

        Returns:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        trans_U = self._U.T
        Q_reindexed_redundant = jnp.einsum(
            "ij,j", trans_U, mass_weight_cartesian_displacement_a0
        )
        Q_redundant = Q_reindexed_redundant[self._arg_sort_index_inverse]
        normal_coor = Q_redundant[6::]
        return normal_coor

    def _convert_normal_to_mass_weight_reindex_U(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to
        mass weighted cartesian displacement coordinates.
        By reindexing transform matrix U.

        Args:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            mass_weight_cartesian_displacement_a0: (15,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the class.
                NOTE: in atomic unit!
        """
        reindex_U = self._U[:, self._arg_sort_index_inverse]
        reindex_U = jax.lax.stop_gradient(reindex_U)
        concise_U = reindex_U[:, 6::]
        mass_weight_cartesian_displacement_a0 = jnp.einsum(
            "ij,j", concise_U, normal_coor
        )
        return mass_weight_cartesian_displacement_a0

    def _convert_mass_weight_to_normal_reindex_U(
        self,
        mass_weight_cartesian_displacement_a0: jax.Array,
    ) -> jax.Array:
        """Convert mass weighted cartesian displacement coordinates to
        normal coordinates.
        By reindexing transform matrix U.

        Args:
            mass_weight_cartesian_displacement_a0: (15,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the class.
                NOTE: in atomic unit!

        Returns:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        trans_U = self._U.T
        trans_U = jax.lax.stop_gradient(trans_U)
        reindex_UT = trans_U[self._arg_sort_index_inverse, :]
        concise_UT = reindex_UT[6::, :]
        normal_coor = jnp.einsum(
            "ij,j", concise_UT, mass_weight_cartesian_displacement_a0
        )
        return normal_coor

    def scaled_normal_to_config_cartesian(
        self,
        scaled_normal: jax.Array,
    ) -> jax.Array:
        """Convert the DIRECT input to KEO and PES, scaled_normal
        into configuration cartesian coordinates

        Args:
            scaled_normal: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.

        Returns:
            configuration_cartesian_coordinates: (5,3) the reshaped
                configuration cartesian coordinates of CH4 with
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]
        """
        pes_weight_cartesian_instance = self.pes_weight_cartesian
        normal_coordinates = self._convert_scaled_normal_to_normal(
            scaled_normal=scaled_normal,
        )
        mass_weight = self._convert_normal_to_mass_weight_reindex_U(
            normal_coor=normal_coordinates,
        )
        mass_weight_angstrom = _convert_a0_to_angstrom(length_in_a0=mass_weight)
        config_cartesian_displacement = pes_weight_cartesian_instance._mwcd2ccd(
            mass_weight_cartesian_displacement=mass_weight_angstrom
        )
        configuration_cartesian_coordinates = (
            pes_weight_cartesian_instance._equilibrium_config
            + config_cartesian_displacement
        ).reshape(5, 3)
        return configuration_cartesian_coordinates

    def config_cartesian_to_scaled_normal(
        self,
        configuration_cartesian_coordinates: jax.Array,
    ) -> jax.Array:
        """Convert configuration cartesian coordinates
        into the DIRECT input to KEO and PES, scaled_x

        Args:
            configuration_cartesian_coordinates: (5,3) the reshaped
                configuration cartesian coordinates of CH4 with
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]

        Returns:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.
        """
        pes_weight_cartesian_instance = self.pes_weight_cartesian
        config_cartesian_coor_flattened = configuration_cartesian_coordinates.reshape(
            -1
        )
        config_cartesian_displacement = (
            config_cartesian_coor_flattened
            - pes_weight_cartesian_instance._equilibrium_config
        )
        mass_weight_cartesian_displacement_angstrom = (
            pes_weight_cartesian_instance._ccd2mwcd(
                configuration_cartesian_displacement=config_cartesian_displacement
            )
        )
        mass_weight_a0 = _convert_angstrom_to_a0(
            length_in_angstrom=mass_weight_cartesian_displacement_angstrom
        )
        normal_coordinates = self._convert_mass_weight_to_normal_reindex_U(
            mass_weight_cartesian_displacement_a0=mass_weight_a0
        )
        scaled_x = self._convert_normal_to_scaled_normal(
            normal_coor=normal_coordinates,
        )
        return scaled_x

    def scaledx_to_config_cartesian(
        self,
        scaled_x: jax.Array,
    ) -> jax.Array:
        """Convert the DIRECT input to KEO and PES, scaled_x
        into configuration cartesian coordinates

        Args:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.

        Returns:
            configuration_cartesian_coordinates: (5,3) the reshaped
                configuration cartesian coordinates of CH4 with
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]
        """
        pes_weight_cartesian_instance = self.pes_weight_cartesian
        normal_coordinates = self._convert_scaled_x_to_normal(
            scaled_x=scaled_x,
        )
        mass_weight = self._convert_normal_to_mass_weight_reindex_U(
            normal_coor=normal_coordinates,
        )
        mass_weight_angstrom = _convert_a0_to_angstrom(length_in_a0=mass_weight)
        config_cartesian_displacement = pes_weight_cartesian_instance._mwcd2ccd(
            mass_weight_cartesian_displacement=mass_weight_angstrom
        )
        configuration_cartesian_coordinates = (
            pes_weight_cartesian_instance._equilibrium_config
            + config_cartesian_displacement
        ).reshape(5, 3)
        return configuration_cartesian_coordinates

    def config_cartesian_to_scaledx(
        self,
        configuration_cartesian_coordinates: jax.Array,
    ) -> jax.Array:
        """Convert configuration cartesian coordinates
        into the DIRECT input to KEO and PES, scaled_x

        Args:
            configuration_cartesian_coordinates: (5,3) the reshaped
                configuration cartesian coordinates of CH4 with
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]

        Returns:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.
        """
        pes_weight_cartesian_instance = self.pes_weight_cartesian
        config_cartesian_coor_flattened = configuration_cartesian_coordinates.reshape(
            -1
        )
        config_cartesian_displacement = (
            config_cartesian_coor_flattened
            - pes_weight_cartesian_instance._equilibrium_config
        )
        mass_weight_cartesian_displacement_angstrom = (
            pes_weight_cartesian_instance._ccd2mwcd(
                configuration_cartesian_displacement=config_cartesian_displacement
            )
        )
        mass_weight_a0 = _convert_angstrom_to_a0(
            length_in_angstrom=mass_weight_cartesian_displacement_angstrom
        )
        normal_coordinates = self._convert_mass_weight_to_normal_reindex_U(
            mass_weight_cartesian_displacement_a0=mass_weight_a0
        )
        scaled_x = self._convert_normal_to_scaled_x(
            normal_coor=normal_coordinates,
        )
        return scaled_x

    def pes(self, normal_coor: jax.Array) -> float:
        """Calculate potential energy from normal coordinates
        with input and output all in a.u.

        Args:
            normal_coor: (9,) the normal coordinates corresponding to
                the order of harmonic frequencies,
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            potential_energy_wavenumber: the corresponding potential energy.
            NOTE: in a.u.! that is, in Eh (Hartree).
        """
        delta_x = self._convert_normal_to_mass_weight(
            normal_coor=normal_coor,
        )
        potential_energy = self._mass_weight_pes(
            mass_weight_cartesian_displacement_a0=delta_x
        )
        return potential_energy

    def pes_in_scaled_x(self, scaled_x: jax.Array) -> float:
        """Calculate potential energy from scaled normal coordinates
        x, with the relationship:

            sqrt(beta/alpha) x_i = Q_i

            Here beta is the conversion coefficient for length from
                the unit used in the main programm, compatible with
                cm^{-1} s.t. the direct output of the kinetic
                function has cm^{-1} unit, to atomic unit(a.u.)
                which measure length in a0 (Bohr). Hence,
                beta = sqrt(219474.63137)
            Here alpha is the omega scaling constants used in the
                main programm for some case convenient for numerical
                stability if the scaled normal coordinates, x,
                is randomly initialized. Typically, if x are
                initialized as zeros, simply set alpha=1.0 would
                not harm numerical stability and leads to a easier
                debugging mood.
            And Q_i is the ith normal coordinate following
                [w1 w2 w2 w3 w3 w3 w4 w4 w4] in a.u.

        Args:
            scaled_x: (9,1) the scaled normal coordinates corresponding to
                the order of harmonic frequencies, used in the main programm,
                NOTE: with a compatible unit to directly give cm^{-1} as called
                in any energy function, for example, kinetic or potential.
                For more details about the unit of scaled_x, see the docstring
                of this function above.

        Returns:
            potential_energy_wavenumber: the corresponding potential energy.
            NOTE: in wavenumber cm^{-1}!
        """
        normal_coor = self._convert_scaled_x_to_normal(scaled_x=scaled_x)
        potential_energy_Hartree = self.pes(normal_coor=normal_coor)
        potential_energy_wavenumber = (
            _convert_Hartree_to_inverse_cm(potential_energy_Hartree) / self.alpha
        )
        return potential_energy_wavenumber

    def pes_in_scaled_normal(self, scaled_normal: jax.Array) -> float:
        """Calculate potential energy from scaled normal coordinates
        , with the relationship:

            sqrt(1/alpha) x_i = Q_i

            Here alpha is the omega scaling constants used in the
                main programm for some case convenient for numerical
                stability if the scaled normal coordinates, x,
                is randomly initialized. Typically, if x are
                initialized as zeros, simply set alpha=1.0 would
                not harm numerical stability and leads to a easier
                debugging mood.
            And Q_i is the ith normal coordinate following
                [w1 w2 w2 w3 w3 w3 w4 w4 w4] in a.u.

        Args:
            scaled_normal: (num_of_modes,1) the scaled normal coordinates
                corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.
        Returns:
            potential_energy_hartree: the corresponding potential energy.
            NOTE: in hartree, not scaled!
        """
        normal_coor = self._convert_scaled_normal_to_normal(scaled_normal=scaled_normal)
        potential_energy_Hartree = self.pes(normal_coor=normal_coor)
        return potential_energy_Hartree

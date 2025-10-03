"""PES for CH5+"""
# %%
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.utils.convert import _convert_inverse_cm_to_hartree
from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_mccoy_jpca_2021_125_5849_5859,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    convert_cartesian_to_sorted_cm,
    load_flax_params,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import get_nn_pes_input
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import McCoyNNPES

jax.config.update("jax_enable_x64", True)


class CH3D2Plus(MoleculeBase):
    """The CH3D2+ Molecule

    Attributes:
        self.pes_origin_func
        self.equilibrium_config
        self.sqrt_atom_masses
        self._arg_sort_index
        self._arg_sort_index_inverse
        self._zeros_in_redundant_normal
        self.trans_normal2mwcd_matrix
        self.w_indices
        self.xalpha

        Above are attributes defined in the base class.

        self._select_potential: the selected potential,
            often from particular artile.
        self.num_of_modes: the number of normal modes.
        self._w_indices_rearrange_index:
            To make omegas from ascending order to the corresponding order
            in ref, which is usually one particular order
            defined in particular paper.

        NOTE: for reindexing Q, see implementation of normal_to_config_cartesian.
    """

    def __init__(
        self,
        select_potential: str,
    ) -> None:
        """Init with select_potential

        Args:
            selected_potential: designating from which article the
                PES is used.


        """
        self._select_potential = select_potential
        if select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            print(f"Select NN-PES from {self._select_potential}.(Fitted near GS)")
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        self.num_of_modes = 12
        self._w_indices_rearrange_index = np.array(range(self.num_of_modes))

    @property
    def pes_origin_func(self) -> Callable[[jax.Array], float]:
        """Original PES

        Returns:
            self._pes_origin_func: the original PES function for
            the specific molecule. Could be a callable
            function in ANY coordinates.
        """
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            params_flax_file = "./neuralvib/molecule/ch5plus/McCoy_NN_PES/params_flax"
            model_flax = McCoyNNPES(out_dims=1)
            params_flax = load_flax_params(filename=params_flax_file)

            def _pes_origin_func(cm: jax.Array) -> float:
                """The origin pes that recieves only coulomb
                matrix as input.
                """
                inference_result = model_flax.apply(params_flax, cm)
                return inference_result.reshape()

            return _pes_origin_func
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )

    @property
    def equilibrium_config(self) -> np.ndarray:
        """Equilibrium configuration

        Returns:
            _equilibrium_config: (deg_of_freedom,) the configuration
                cartesian coordinates for each atom, flattened, in a.u.
            For CH5+, specificlly,
                eq_conf = np.array(
                    [
                        Cx,Cy,Cz,
                        H1x,H1y,H1z,
                        H2x,H2y,H2z,
                        H3x,H3y,H3z,
                        H4x,H4y,H4z,
                        H5x,H5y,H5z,
                    ]
                )
                in which the coordinates are the atoms' equilibrium
                position, setting carbon as the origin of the coordinate
                system.
                NOTE: the order of the hydrogen atoms are
                the same as in  J. Chem. Phys. 121, 4105-4116(2004),
                Brown, McCoy, Braams, Jin and Bowman.
        """
        # _equilibrium_config = equilibrium_bowman_jcp_121_4105_4116_2004()
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            _equilibrium_config = equilibrium_mccoy_jpca_2021_125_5849_5859()
        _equilibrium_config = jax.lax.stop_gradient(_equilibrium_config)
        return _equilibrium_config

    @property
    def sqrt_atom_masses(self) -> np.ndarray:
        """Atom masses

        Returns:
            _sqrt_atom_masses: (deg_of_greedom,) the atom masses mapping with
                the equilibrium_config. Specificlly, for CH5+,
                sqrt_atom_masses = np.sqrt(
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
        """

        mass_c = 12.011 * 1836.152673
        mass_h_1 = 1.0079 * 1836.152673
        mass_h_2 = 1.0079 * 1836.152673
        mass_h_3 = 1.0079 * 1836.152673
        mass_h_4 = 2 * 1.0079 * 1836.152673
        mass_h_5 = 2 * 1.0079 * 1836.152673

        _sqrt_atom_masses = np.sqrt(
            np.array(
                [
                    mass_c,
                    mass_c,
                    mass_c,
                    mass_h_1,
                    mass_h_1,
                    mass_h_1,
                    mass_h_2,
                    mass_h_2,
                    mass_h_2,
                    mass_h_3,
                    mass_h_3,
                    mass_h_3,
                    mass_h_4,
                    mass_h_4,
                    mass_h_4,
                    mass_h_5,
                    mass_h_5,
                    mass_h_5,
                ]
            )
        )
        return _sqrt_atom_masses

    @property
    def _arg_sort_index(self) -> np.ndarray:
        """Reindexing Q arg sort index

        Returns:
            _sort_index: (deg_of_freedom, ) the arg_sort result
                for reindexing Q.
                e.g. _arg_sort_index: from [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]:
                    the common notation of different harmonic frequencies,
                to [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6]
                    in the ascending order of harmonic frequencies,
                    s.t. w7 <= w8 <= w9 <= w2 <= w3 <= w1 <= w4 <= w5 <= w6.
        NOTE: The w_indices are chosen the same as in
            J. Phys. Chem. A 2006, 110, 1569-1574
            TABLE 1, which is in ascending order!
        """
        num_of_modes = self.num_of_modes
        zeros_num = self._zeros_in_redundant_normal
        sort_num = num_of_modes + zeros_num
        _sort_index = np.array(range(sort_num))
        return _sort_index

    @property
    def _arg_sort_index_inverse(self) -> np.ndarray:
        """Inverse reindexing Q arg sort index

        Returns:
            _sort_index_inverse: (deg_of_freedom, ) the inverse arg_sort result
                for reindexing Q.
                e.g. _arg_sort_index_inverse: from
                    [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6]
                    in the ascending order of harmonic frequencies,
                    s.t. w7 <= w8 <= w9 <= w2 <= w3 <= w1 <= w4 <= w5 <= w6.
                to [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]:
                    the common notation of different harmonic frequencies,
        """
        num_of_modes = self.num_of_modes
        zeros_num = self._zeros_in_redundant_normal
        sort_num = num_of_modes + zeros_num
        _sort_index_inverse = np.array(range(sort_num))
        return _sort_index_inverse

    @property
    def _zeros_in_redundant_normal(self) -> int:
        """Number of zeros to add in the redundant Q

        Returns:
            self.__zeros_in_redundant_normal: the zeros to append in redundant normal
                coordinates. This would be 6 for non-linear molecule and
                5 for linear molecule.
        """
        zeros_num = 6
        return zeros_num

    @property
    def trans_normal2mwcd_matrix(self) -> jax.Array:
        """The transformation matrix U from
        Redundant normal coordinates to
        mass-weighted cartesian displacement coordinates.

        Returns:
            _trans_normal2mwcd_matrix: (deg_of_freedom,deg_of_freedom)
                the transformation matrix
                that transforms (redundant) normal coordinates to
                mass-weighted cartesian displacement coordinates.k.
        """
        pes_mwcd = self.pes_mwcd
        deg_of_freedom = self.num_of_modes + self._zeros_in_redundant_normal
        x_e = jnp.zeros(deg_of_freedom)

        _trans_normal2mwcd_matrix = self.get_transfer_matrix_from_pes_ad(
            pes=pes_mwcd,
            x_e=x_e,
        )
        return _trans_normal2mwcd_matrix

    @property
    def w_indices(self) -> np.ndarray:
        """The w_indices of the molecule.
        NOTE: in a.u.

        Returns:
            _w_indices: the harmonic frequencies omegas
                of the molecule. If degeneracy occurs,
                the ws are indexed sequentially. For example,
                if w1 is two-fold degenerated, then they are
                indexed as w1,w2.
        """
        pes_mwcd = self.pes_mwcd
        deg_of_freedom = self.num_of_modes + self._zeros_in_redundant_normal
        x_e = jnp.zeros(deg_of_freedom)
        _w_indices = self.get_w_indices_from_pes_ad(
            pes=pes_mwcd,
            x_e=x_e,
            w_indices_rearrange_index=self._w_indices_rearrange_index,
        )

        # internal diagonalization check
        # this gurantees that the harmonic
        # frequencies are ordered as we expected.
        # __expected_eigval = jnp.array(
        #     [
        #         -1.21544893e-04,
        #         -3.84327296e-06,
        #         -2.07449496e-06,
        #         -2.70715946e-21,
        #         -7.16693026e-22,
        #         1.26174388e-21,
        #         3.40375836e-06,
        #         6.82848165e-06,
        #         1.15722764e-05,
        #         1.58500788e-05,
        #         2.39191134e-05,
        #         2.79442802e-05,
        #         3.64900833e-05,
        #         4.26795151e-05,
        #         5.96197379e-05,
        #         1.78176401e-04,
        #         2.22424744e-04,
        #         2.50583145e-04,
        #     ],
        #     dtype=jnp.float64,
        # )
        # _, __eigval = normal_to_cartesian_displacement(pes=pes_mwcd, x_e=x_e)
        # if not jnp.allclose(__eigval, __expected_eigval):
        #     raise ValueError(
        #         "the expected behaviour of eigh may change and a "
        #         "reinspection to arg_sort_index is needed!"
        #     )
        # NOTE: for CH5+ this may be skipped since
        # the w_indices in Bowman's PES paper
        # J. Phys. Chem. A 2006, 110, 1569-1574
        # are in ascending order!

        return _w_indices

    @property
    def xalpha(self) -> float:
        """The scaled normal coordinates scaling factor.

        Returns:
            self._xalpha: the scaling factor of normal coordinates
                s.t. sqrt(1/alpha) x_i = Q_i, where x_i is the
                scaled_normal directly used in the programm
                and Q_i is the physical normal coordinates.
        """
        return self._xalpha

    @xalpha.setter
    def xalpha(self, xalpha: float) -> None:
        """Set xalpha"""
        self._xalpha = xalpha

    def convert_mwcd_to_pes_origin_func_input(
        self,
        mass_weight_cartesian_displacement_a0: jax.Array,
    ) -> jax.Array:
        """Convert the mass weighted cartesian displacement (mwcd)
        coordinates in a0 (atomic unit) into the input of the original PES function.
        NOTE: MUST be rewritten in subclass!

        Args:
            mass_weight_cartesian_displacement_a0: (dim * num_of_atoms,) the flattened
                mass weight cartesian displacement coordinates,
                in a specific order. For example, for CH4:
            (C_x C_y C_z H_1x H_1y H_1z H_2x H_2y H_2z H_3x H_3y H_3z H_4x H_4y H_4z)
                NOTE: in atomic unit!

        Returns:
            pes_origin_func_input: the DIRECT original PES function input.
        """
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            config_cartesian_displacement = self.mwcd2ccd(
                mass_weight_cartesian_displacement=mass_weight_cartesian_displacement_a0
            )
            config_cartesian_coor_flattened = (
                jnp.array(self.equilibrium_config) + config_cartesian_displacement
            )
            sorted_coulomb_matrix = convert_cartesian_to_sorted_cm(
                cartesian_coors=config_cartesian_coor_flattened
            )
            pes_origin_func_input = get_nn_pes_input(sorted_cm=sorted_coulomb_matrix)
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        return pes_origin_func_input

    def convert_pes_origin_func_energy_to_hartree(
        self,
        pes_origin_func_energy: float,
    ) -> float:
        """Convert the energy that acts as direct output
        of the original PES to Hartree.
        NOTE: MUST be rewritten in subclass!

        Args:
            pes_origin_func_energy: the direct output of
                the original PES function.

        Returns:
            potential_energy_in_hartree: the corresponding
                potential energy in hartree.
        """
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            coefficient = _convert_inverse_cm_to_hartree(1.0)
            potential_energy_in_hartree = coefficient * pes_origin_func_energy
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        return potential_energy_in_hartree

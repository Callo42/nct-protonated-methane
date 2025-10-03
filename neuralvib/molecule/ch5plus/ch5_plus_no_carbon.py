"""PES for CH5+"""

# %%
import re
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.JBB_Full_PES.convert_coors import (
    config_cartesian2jbb_cartesian_input_xn,
)
from neuralvib.molecule.ch5plus.JBB_Full_PES.jbbjax import (
    jbbexternal,
    jbbexternal_withreject,
    jbbjax_cartesian,
)
from neuralvib.molecule.ch5plus.pyscf_PES.CCSD import (
    pyscf_ccsd_jax_pure_cartesian,
    pyscf_ccsd_pes_cartesian,
)
from neuralvib.molecule.ch5plus.pyscf_PES.dft import (
    pyscf_dft_jax_pure_cartesian,
    pyscf_dft_pes_cartesian,
)
from neuralvib.molecule.ch5plus.pyscf_PES.mp2 import (
    pyscf_mp2_jax_pure_cartesian,
    pyscf_mp2_pes_cartesian,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.convert import _convert_inverse_cm_to_hartree
from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_mccoy_jpca_2021_125_5849_5859,
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    convert_cartesian_to_sorted_cm,
    load_flax_params,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import get_nn_pes_input
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import McCoyNNPES
from neuralvib.molecule.ch5plus.pyscf_PES.hf import (
    pyscf_hf_jax_io_cartesian,
    pyscf_hf_jax_pure_cartesian,
    pyscf_hf_pes_cartesian,
)

jax.config.update("jax_enable_x64", True)


class CH5PlusNoCarbon(MoleculeBase):
    """The CH5+ Molecule
    without carbon atom
    hence is supposed that carbon is always
    the origin of the coordinates.

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
        self.external_pes

        Above are attributes defined in the base class.

        self._select_potential: the selected potential,
            often from particular artile.
        self.select_potential: the selected potential,
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
        self.particles: tuple = ("H", "H", "H", "H", "H")
        self.particle_mass: dict = {
            "H": 1.0079 * 1836.152673,
        }
        self._select_potential = select_potential
        if select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            print(f"Select NN-PES from {self._select_potential}.(Fitted near GS)")
        elif select_potential == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib":
            print(f"Select JBB Full PES from {self._select_potential}.")
        elif select_potential == "J.Phys.Chem.A2006,110,1569-1574":
            print(f"Select JBB Full PES from {self._select_potential}.")
        elif select_potential == "PySCF.HF.pure":
            print("Select PySCF Hartree Fock pure_callback as PES.")
        elif select_potential == "PySCF.HF.io":
            print("Select PySCF Hartree Fock io_callback as PES.")
        elif select_potential == "PySCF.MP2.pure":
            print("Select PySCF MP2 pure_callback as PES.")
        elif select_potential == "PySCF.CCSD.pure":
            print("Select PySCF CCSD pure_callback as PES.")
        elif select_potential == "PySCF.DFT.pure":
            print("Select PySCF DFT pure_callback as PES.")
        elif select_potential == "External.PySCF.HF.Joblib":
            print("Select PySCF HF External Joblib Parallel as PES.")
        elif select_potential == "External.PySCF.MP2.Joblib":
            print("Select PySCF MP2 External Joblib Parallel as PES.")
        elif select_potential == "External.PySCF.DFT.Joblib":
            print("Select PySCF DFT External Joblib Parallel as PES.")
        elif select_potential == "External.PySCF.CCSD.Joblib":
            print("Select PySCF CCSD External Joblib Parallel as PES.")

        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        self.num_of_modes = 12
        self._w_indices_rearrange_index = np.array(range(self.num_of_modes))

    @property
    def pes_origin_func(self) -> Callable[[np.ndarray], float]:
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

            def _nn_pes_origin_func(cm: np.ndarray) -> float:
                """The origin pes that recieves only coulomb
                matrix as input.
                Args:
                    cm: the Coulomb Matrix, see paper.
                """
                inference_result = model_flax.apply(params_flax, cm)
                inference_result = inference_result.reshape()
                inference_result = jnp.where(
                    inference_result < 0, 2000000, inference_result
                )
                return inference_result

            _pes_origin_func = _nn_pes_origin_func

        elif self._select_potential == "J.Phys.Chem.A2006,110,1569-1574":
            _pes_origin_func = jbbjax_cartesian
        elif (
            self._select_potential == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib"
        ):
            # _pes_origin_func = jbbexternal_withreject
            _pes_origin_func = jbbexternal
        elif self._select_potential == "PySCF.HF.pure":
            _pes_origin_func = pyscf_hf_jax_pure_cartesian
        elif self._select_potential == "PySCF.HF.io":
            _pes_origin_func = pyscf_hf_jax_io_cartesian
        elif self._select_potential == "PySCF.MP2.pure":
            _pes_origin_func = pyscf_mp2_jax_pure_cartesian
        elif self._select_potential == "PySCF.CCSD.pure":
            _pes_origin_func = pyscf_ccsd_jax_pure_cartesian
        elif self._select_potential == "PySCF.DFT.pure":
            _pes_origin_func = pyscf_dft_jax_pure_cartesian
        elif self._select_potential == "External.PySCF.HF.Joblib":
            _pes_origin_func = pyscf_hf_pes_cartesian
        elif self._select_potential == "External.PySCF.MP2.Joblib":
            _pes_origin_func = pyscf_mp2_pes_cartesian
        elif self._select_potential == "External.PySCF.DFT.Joblib":
            _pes_origin_func = pyscf_dft_pes_cartesian
        elif self._select_potential == "External.PySCF.CCSD.Joblib":
            _pes_origin_func = pyscf_ccsd_pes_cartesian
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        return _pes_origin_func

    @property
    def equilibrium_config(self) -> np.ndarray:
        """Equilibrium configuration
        NOTE: now only act as the reference point
        of the basis in config flow calculation.

        Returns:
            _equilibrium_config: (deg_of_freedom,) the configuration
                cartesian coordinates for each atom, flattened, in a.u.
            For CH5+, specificlly,
                eq_conf = np.array(
                    [
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
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            _equilibrium_config = equilibrium_mccoy_jpca_2021_125_5849_5859()[3:]
            # _equilibrium_config = equilibrium_bowman_jpca_2006_110_1569_1574()
        elif (
            self._select_potential == "J.Phys.Chem.A2006,110,1569-1574"
            or self._select_potential
            == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib"
        ):
            _equilibrium_config = equilibrium_bowman_jpca_2006_110_1569_1574()[3:]
            _equilibrium_config = np.zeros_like(_equilibrium_config)
            # TODO: change to reference point
            # actually reference point
            # _equilibrium_config = saddle_c2v_bowman_jpca_2006_110_1569_1574()[3:]
        elif (
            self._select_potential == "PySCF.HF.pure"
            or self._select_potential == "PySCF.HF.io"
            or self._select_potential == "PySCF.MP2.pure"
            or self._select_potential == "PySCF.CCSD.pure"
            or self._select_potential == "PySCF.DFT.pure"
            or self._select_potential == "External.PySCF.HF.Joblib"
            or self._select_potential == "External.PySCF.MP2.Joblib"
            or self._select_potential == "External.PySCF.DFT.Joblib"
            or self._select_potential == "External.PySCF.CCSD.Joblib"
        ):
            _equilibrium_config = equilibrium_bowman_jpca_2006_110_1569_1574()[3:]
        # _equilibrium_config = jax.lax.stop_gradient(_equilibrium_config)
        _equilibrium_config = np.zeros_like(_equilibrium_config)
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

        mass_h_1 = 1.0079 * 1836.152673
        mass_h_2 = 1.0079 * 1836.152673
        mass_h_3 = 1.0079 * 1836.152673
        mass_h_4 = 1.0079 * 1836.152673
        mass_h_5 = 1.0079 * 1836.152673

        _sqrt_atom_masses = np.sqrt(
            np.array(
                [
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

    @property
    def _zeros_in_redundant_normal(self) -> int:
        """Number of zeros to add in the redundant Q

        Returns:
            self.__zeros_in_redundant_normal: the zeros to append in redundant normal
                coordinates. This would be 6 for non-linear molecule and
                5 for linear molecule.
        """

    @property
    def trans_normal2mwcd_matrix(self) -> np.ndarray:
        """The transformation matrix U from
        Redundant normal coordinates to
        mass-weighted cartesian displacement coordinates.

        Returns:
            _trans_normal2mwcd_matrix: (deg_of_freedom,deg_of_freedom)
                the transformation matrix
                that transforms (redundant) normal coordinates to
                mass-weighted cartesian displacement coordinates.k.

        NOTE: the trans_normal2mwcd_matrix for CH5+ is first
        computed by NN-PES, then serialized and reloaded
        each time in future CH5+ molecule instance!
        This is really a slightly compromising choice
        since in fact the flow should be able to learn
        the accurate transfer matrix, so a initial transfer
        matrix not so precise may be acceptable.

        For example, if directly calculate this transfer matrix
        utilizing AD from original pes:
        ```python
        pes_mwcd = self.pes_mwcd
        deg_of_freedom = self.num_of_modes + self._zeros_in_redundant_normal
        x_e = jnp.zeros(deg_of_freedom)

        _trans_normal2mwcd_matrix = self.get_transfer_matrix_from_pes_ad(
            pes=pes_mwcd,
            x_e=x_e,
        )
        ```
        """

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

    @property
    def external_pes(self) -> bool:
        """Indication of the PES type.

        Returns:
            self._external_pes: if false, the corresponding
                coordinate transformation for PES would
                use jax functions, otherwise would use a
                pure numpy version.
        """
        match_external = re.search(r"External", self._select_potential, re.IGNORECASE)
        if match_external:
            # External PES, using np version
            return True
        else:
            # jax based PES (original or callback)
            # use jax version
            return False

    @property
    def select_potential(self) -> str:
        """The selected potential.

        Returns:
            self._select_potential: the selected potential.
        """
        return self._select_potential

    def convert_mwcd_to_pes_origin_func_input(
        self,
        mass_weight_cartesian_displacement_a0: np.ndarray,
    ) -> np.ndarray | jax.Array:
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

    def convert_config_cartesian_to_pes_origin_func_input(
        self,
        cartesian_coors: np.ndarray,
    ) -> np.ndarray | jax.Array:
        """Convert the configuration cartesian coordinates
        in a0 (atomic unit) into the input of the original PES function.
        NOTE: MUST be rewritten in subclass!

        Args:
            cartesian_coors: (5,dim,) the
                cartesian coordinates in a.u.
                in a specific order. For example, for CH4:
                NOTE: in atomic unit!

        Returns:
            pes_origin_func_input: the DIRECT original PES function input.
        """
        carbon = np.zeros(3)
        config_cartesian_coor_flattened = cartesian_coors.reshape(-1)
        config_cartesian_coor_flattened = jnp.concatenate(
            (carbon, config_cartesian_coor_flattened)
        )
        if self._select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            sorted_coulomb_matrix = convert_cartesian_to_sorted_cm(
                cartesian_coors=config_cartesian_coor_flattened
            )
            pes_origin_func_input = get_nn_pes_input(sorted_cm=sorted_coulomb_matrix)
        elif (
            self._select_potential == "J.Phys.Chem.A2006,110,1569-1574"
            or self._select_potential
            == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib"
        ):
            pes_origin_func_input = config_cartesian2jbb_cartesian_input_xn(
                cartesian_coors=config_cartesian_coor_flattened
            )
        elif (
            self._select_potential == "PySCF.HF.pure"
            or self._select_potential == "PySCF.HF.io"
            or self._select_potential == "PySCF.MP2.pure"
            or self._select_potential == "PySCF.CCSD.pure"
            or self._select_potential == "PySCF.DFT.pure"
            or self._select_potential == "External.PySCF.HF.Joblib"
            or self._select_potential == "External.PySCF.MP2.Joblib"
            or self._select_potential == "External.PySCF.DFT.Joblib"
            or self._select_potential == "External.PySCF.CCSD.Joblib"
        ):
            pes_origin_func_input = config_cartesian_coor_flattened.reshape(6, 3)
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
        elif (
            self._select_potential == "J.Phys.Chem.A2006,110,1569-1574"
            or self._select_potential
            == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib"
        ):
            coefficient = 1.0
            # coefficient = _convert_inverse_cm_to_hartree(1.0)
            potential_energy_in_hartree = coefficient * pes_origin_func_energy
        elif (
            self._select_potential == "PySCF.HF.pure"
            or self._select_potential == "PySCF.HF.io"
            or self._select_potential == "PySCF.MP2.pure"
            or self._select_potential == "PySCF.CCSD.pure"
            or self._select_potential == "PySCF.DFT.pure"
            or self._select_potential == "External.PySCF.HF.Joblib"
            or self._select_potential == "External.PySCF.MP2.Joblib"
            or self._select_potential == "External.PySCF.DFT.Joblib"
            or self._select_potential == "External.PySCF.CCSD.Joblib"
        ):
            coefficient = 1.0
            potential_energy_in_hartree = coefficient * pes_origin_func_energy
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        return potential_energy_in_hartree

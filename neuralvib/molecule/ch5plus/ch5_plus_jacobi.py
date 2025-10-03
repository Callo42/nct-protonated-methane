"""PES for CH5+"""

import re
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch5plus.JBB_Full_PES.jbbjax import (
    jbbexternal,
    jbbjax_cartesian,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.utils.convert import _convert_inverse_cm_to_hartree
from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    load_flax_params,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import McCoyNNPES


jax.config.update("jax_enable_x64", True)


def config2jacobi(
    coors: np.ndarray | jax.Array,
) -> np.ndarray | jax.Array:
    """Convert cartesian coordinates to jacobi coordinates.
    Args:
        coors: (6,3) the cartesian coordinates in a.u.
            NOTE: Currently FORCE the order of jacobi coors
                s.t. the jacobi coors are constructed from
                the atoms in lab frame following the order
                H1,H2,H3,H4,H5,C,
                which is, the jacobi coor r1 is x_{H1} - x_{H2},
                r2 = (m_H x_{H1} + m_H x_{H2})/(m_H + m_H) - x_{H3}, etc.
    Returns:
        jacobi_coors: (5,3) the jacobi vectors in a.u.
            NOTE: Currently FORCE the order of jacobi coors
                s.t. the jacobi coors are constructed from
                the atoms in lab frame following the order
                H1,H2,H3,H4,H5,C,
                which is, the jacobi coor r1 is x_{H1} - x_{H2},
                r2 = (m_H x_{H1} + m_H x_{H2})/(m_H + m_H) - x_{H3}, etc.
    """
    assert coors.shape == (6, 3)
    xc = coors[0]
    x1 = coors[1]
    x2 = coors[2]
    x3 = coors[3]
    x4 = coors[4]
    x5 = coors[5]
    r1 = x1 - x2
    r2 = (x1 + x2) / 2 - x3
    r3 = (x1 + x2 + x3) / 3 - x4
    r4 = (x1 + x2 + x3 + x4) / 4 - x5
    r5 = (x1 + x2 + x3 + x4 + x5) / 5 - xc
    jacobi_coors = jnp.array([r1, r2, r3, r4, r5])
    return jacobi_coors


class CH5PlusJacobi(MoleculeBase):
    """The CH5+ Molecule
    at center of mass (CoM) frame under jacobi coors.
    Here we use the 15 jacobi vectors in rectilinear frame.

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
        NOTE: Currently FORCE the order of jacobi coors
            s.t. the jacobi coors are constructed from
            the atoms in lab frame following the order
            H1,H2,H3,H4,H5,C,
            which is, the jacobi coor r1 is x_{H1} - x_{H2},
            r2 = (m_H x_{H1} + m_H x_{H2})/(m_H + m_H) - x_{H3}, etc.

        """
        # a.u.
        self.mass_hydrogen = 1.0079 * 1836.152673
        self.mass_carbon = 12.011 * 1836.152673

        # the "relative particle defined in Jacobi Coors"
        # NOTE the reduced mass depends on the "order" of
        # constructing jacobi coors! And should not be changed! See docstring.
        self.particles: tuple = ("R1", "R2", "R3", "R4", "R5")
        self.particle_mass: dict = {
            "R1": self.mass_hydrogen / 2,
            "R2": 2 * self.mass_hydrogen / 3,
            "R3": 3 * self.mass_hydrogen / 4,
            "R4": 4 * self.mass_hydrogen / 5,
            "R5": 5
            * self.mass_hydrogen
            * self.mass_carbon
            / (5 * self.mass_hydrogen + self.mass_carbon),
        }

        self._select_potential = select_potential
        if select_potential == "J.Phys.Chem.A2021,125,5849-5859":
            print(f"Select NN-PES from {self._select_potential}.(Fitted near GS)")
        elif select_potential == "External.J.Phys.Chem.A2006,110,1569-1574.Joblib":
            print(f"Select JBB Full PES from {self._select_potential}.")
        elif select_potential == "J.Phys.Chem.A2006,110,1569-1574":
            print(f"Select JBB Full PES from {self._select_potential}.")
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
        _equilibrium_config = saddle_c2v_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        _equilibrium_config = config2jacobi(_equilibrium_config)
        return _equilibrium_config

    @property
    def c2v_config_in_jacobi(self) -> np.ndarray:
        """C2V configuration in Jacobi coordinates"""
        _config = saddle_c2v_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        _config = config2jacobi(_config)
        return _config

    @property
    def sqrt_atom_masses(self) -> np.ndarray:
        """"""

    @property
    def _arg_sort_index(self) -> np.ndarray:
        """"""

    @property
    def _arg_sort_index_inverse(self) -> np.ndarray:
        """"""

    @property
    def _zeros_in_redundant_normal(self) -> int:
        """"""

    @property
    def trans_normal2mwcd_matrix(self) -> np.ndarray:
        """"""

    @property
    def w_indices(self) -> np.ndarray:
        """"""

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
        """"""

    def convert_config_cartesian_to_pes_origin_func_input(
        self,
        cartesian_coors: np.ndarray,
    ) -> np.ndarray | jax.Array:
        """"""

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
        else:
            raise NotImplementedError(
                f"Seleted PES {self._select_potential} not implemented!"
            )
        return potential_energy_in_hartree

    def convert_jacobi2jbb_cartesian_input_xn(
        self,
        coors: jax.Array | np.ndarray,
    ) -> jax.Array | np.ndarray:
        """Convert jacobi coordinates (rectilinear)
        to cartesian coordinates xn for CH5+
        NOTE: for JBB Full PES input only!

        Args:
            coors: (15,) the FLATTENED jacobi vectors in a.u.
            NOTE: Currently FORCE the order of jacobi coors
                s.t. the jacobi coors are constructed from
                the atoms in lab frame following the order
                H1,H2,H3,H4,H5,C,
                which is, the jacobi coor r1 is x_{H1} - x_{H2},
                r2 = (m_H x_{H1} + m_H x_{H2})/(m_H + m_H) - x_{H3}, etc.

        Returns:
            xn: (3,6)  is the Cartesian coordinates for
            six atoms in order of
            C H H H H H. (in bohr)
            with C as origin.
            NOTE:the recieved getpot recieves xn which is in shape(3,6)!
                see getpot.f for details.
        """
        coors = coors.reshape(5, 3)
        r1 = coors[0]
        r2 = coors[1]
        r3 = coors[2]
        r4 = coors[3]
        r5 = coors[4]

        last_coeff = (
            -5
            * self.mass_hydrogen
            / (5 * self.mass_hydrogen * (self.mass_carbon + 1) + self.mass_carbon**2)
        )
        _carbon = last_coeff * r5
        _h_5 = (5 * r5 - 4 * r4) / 5 + _carbon
        _h_4 = (4 * r4 - 3 * r3) / 4 + _h_5
        _h_3 = (3 * r3 - 2 * r2) / 3 + _h_4
        _h_2 = (2 * r2 - r1) / 2 + _h_3
        _h_1 = r1 + _h_2

        config_cart = jnp.array(
            [
                _carbon,
                _h_1,
                _h_2,
                _h_3,
                _h_4,
                _h_5,
            ]
        )
        cart_c_origin = config_cart - config_cart[0]
        xn = cart_c_origin.T
        return xn

    def pes(self, coors: np.ndarray | jax.Array) -> float:
        """The pes that recieves jacobi coors
            Args:
                coors: (15,) the FLATTENED jacobi vectors in a.u.
        NOTE: Currently FORCE the order of jacobi coors
            s.t. the jacobi coors are constructed from
            the atoms in lab frame following the order
            H1,H2,H3,H4,H5,C,
            which is, the jacobi coor r1 is x_{H1} - x_{H2},
            r2 = (m_H x_{H1} + m_H x_{H2})/(m_H + m_H) - x_{H3}, etc.
        """
        assert self._select_potential == "J.Phys.Chem.A2006,110,1569-1574"
        # currently only implemented for coor conversion for JBB PES.
        pes_input = self.convert_jacobi2jbb_cartesian_input_xn(coors)
        pot_energy_origin = self.pes_origin_func(pes_input)
        pot_energy_hartree = self.convert_pes_origin_func_energy_to_hartree(
            pes_origin_func_energy=pot_energy_origin
        )
        return pot_energy_hartree

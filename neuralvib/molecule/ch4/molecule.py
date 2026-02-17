"""CH4 molecule entry point for Cartesian workflow."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.molecule.ch4.pes import CH4PotentialCartesian
from neuralvib.molecule.ch4.trans2normal import _normal_to_cartesian_displacement
from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.utils.convert import _convert_angstrom_to_a0, convert_inverse_cm_to_hartree

jax.config.update("jax_enable_x64", True)


class CH4(MoleculeBase):
    """Methane molecule using the CCSD(T)/cc-pVTZ PES (JCP 102, 254-261, 1995)."""

    def __init__(self, select_potential: str) -> None:
        self.particles: tuple[str, ...] = ("C", "H", "H", "H", "H")
        self.particle_mass: dict[str, float] = {
            "C": 12.011 * 1836.152673,
            "H": 1.0079 * 1836.152673,
        }
        self.num_of_modes = 9
        self._zeros_in_redundant_normal = 6  # non-linear molecule
        self._select_potential = select_potential
        if select_potential not in (
            "J.Chem.Phys.102,254-261(1995)",
        ):
            raise NotImplementedError(
                f"Selected PES '{select_potential}' not implemented for CH4."
            )

        self._pes_cartesian = CH4PotentialCartesian(select_potential=select_potential)
        eq_cart_angstrom = np.array(self._pes_cartesian._equilibrium_config)
        self._equilibrium_config = _convert_angstrom_to_a0(eq_cart_angstrom)
        self._arg_sort_index = jnp.array(
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 7, 8, 6, 9, 10, 11],
            dtype=jnp.int32,
        )
        self._arg_sort_index_inverse = jnp.array(
            [0, 1, 2, 3, 4, 5, 11, 9, 10, 12, 13, 14, 6, 7, 8],
            dtype=jnp.int32,
        )
        self._xalpha = 1.0

    @property
    def pes_origin_func(self):
        return self._pes_cartesian.pes

    @property
    def equilibrium_config(self) -> np.ndarray:
        """By definition it is under CoM"""
        return np.array(self._equilibrium_config, dtype=np.float64)

    @property
    def sqrt_atom_masses(self) -> np.ndarray:
        """"""
        mass_C = self.particle_mass["C"]
        mass_H_1 = self.particle_mass["H"]
        mass_H_2 = self.particle_mass["H"]
        mass_H_3 = self.particle_mass["H"]
        mass_H_4 = self.particle_mass["H"]
        return np.sqrt(
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

    @property
    def _arg_sort_index(self) -> np.ndarray:  # type: ignore[override]
        return self.__arg_sort_index

    @_arg_sort_index.setter
    def _arg_sort_index(self, value):
        self.__arg_sort_index = value

    @property
    def _arg_sort_index_inverse(self) -> np.ndarray:  # type: ignore[override]
        return self.__arg_sort_index_inverse

    @_arg_sort_index_inverse.setter
    def _arg_sort_index_inverse(self, value):
        self.__arg_sort_index_inverse = value

    @property
    def _zeros_in_redundant_normal(self) -> int:  # type: ignore[override]
        return self.__zeros_in_redundant_normal

    @_zeros_in_redundant_normal.setter
    def _zeros_in_redundant_normal(self, value: int):
        self.__zeros_in_redundant_normal = value

    @property
    def trans_normal2mwcd_matrix(self) -> np.ndarray:
        """"""

    @property
    def w_indices(self) -> np.ndarray:
        w_indices_cm_inv = np.array(
            [
                3035.57114505,
                1571.2439802,
                1571.2439802,
                3153.61820892,
                3153.61820892,
                3153.61820892,
                1343.53679863,
                1343.53679863,
                1343.53679863,
            ],
            dtype=np.float64,
        )
        return convert_inverse_cm_to_hartree(w_indices_cm_inv)

    @property
    def xalpha(self) -> float:
        return self._xalpha

    @xalpha.setter
    def xalpha(self, xalpha: float) -> None:
        self._xalpha = xalpha

    @property
    def external_pes(self) -> bool:
        return False

    @property
    def ms(self) -> np.ndarray:
        masses = []
        for particle in self.particles:
            masses.append([self.particle_mass[particle]] * 3)
        return np.array(masses, dtype=np.float64)

    def convert_mwcd_to_pes_origin_func_input(
        self, mass_weight_cartesian_displacement_a0: np.ndarray | jax.Array
    ) -> np.ndarray | jax.Array:
        return mass_weight_cartesian_displacement_a0

    def convert_config_cartesian_to_pes_origin_func_input(
        self, cartesian_coors: np.ndarray | jax.Array
    ) -> np.ndarray | jax.Array:
        displacement = jnp.array(cartesian_coors).reshape(-1) - self.equilibrium_config
        mwcd = self.ccd2mwcd(displacement)
        return mwcd

    def convert_pes_origin_func_energy_to_hartree(
        self, pes_origin_func_energy: float
    ) -> float:
        return jnp.array(pes_origin_func_energy, dtype=jnp.float64)

"""Molecule PES Base"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


def normal_to_cartesian_displacement(
    pes: Callable,
    x_e: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """calculate the transformation matrix U s.t.
    x = UQ
    Here x is the mass-weighted cartesian displacement coordinated
    and Q is the reindexed rectilinear normal coordinates (redundant),
    all in flattened form.

    NOTE: here Q is REDUNDANT means that 6(or 5) of the Qs
    would be mannually set to zero in real calculation. Here U has the
    shape (deg_of_freedom,deg_of_freedom) is simply for simplicity of
    making matrix inversion.

    NOTE: here Q is REINDEXED from the redundant Q used in calculating
    potential energy in normal coordinates, s.t. they would give
    right order of x:

    One example of using the returns of this function to calculate
    potential energy:

    Q_reindexed = Q_origin_redundant[arg_sort_index]
    delta_x = jnp.einsum("ij,j",U,Q_reindexed)
    potential_energy = pes(delta_x)

    Args:
        pes: the function that take x as input, returns potential
            energy. Specificly this pes is the pes
            under mass-weighted cartesian displacement coordinates.
            NOTE: input and output of pes are all in a.u.
        x_e: (deg_of_freedom, ) the x at equilibrium configuration.
            Specifically x_e are all zeros since the pes ONLY
            recieves mass-weighted cartesian displacement coordinates
            and in this case the x at equilibrium configuration
            are all zeros.

    Returns:
        U: (deg_of_freedom,deg_of_freedom) the transformation matrix
            that transforms (redundant) normal coordinates to
            mass-weighted cartesian displacement coordinates.
        eigval: (deg_of_freedom,) the eigval from diagnolizing
            Hessian. Should be used in further eigenvalue check,
            for example,
            ```
            if not jnp.allclose(eigval, expected_eigval):
                raise ValueError(
                    "the expected behaviour of eigh may change and a "
                    "reinspection to arg_sort_index is needed!"
                )
            ```
    """
    hessian = jax.hessian(pes)(x_e)
    eigval, eigvectors = jnp.linalg.eigh(hessian)  # since pes hermitian
    U = eigvectors
    return U, eigval


class MoleculeBase(ABC):
    """Molecular PES base, including in cartesian coordinates
    and normal coordinates.

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

        NOTE: for reindexing Q, see implementation of normal_to_config_cartesian.
    """

    @property
    @abstractmethod
    def pes_origin_func(self) -> Callable[[np.ndarray], float]:
        """Original PES
        NOTE: the input and output of original pes
        are the same as the original pes function,
        NOT necessarily in a.u.

        Returns:
            self._pes_origin_func: the original PES function for
            the specific molecule. Could be a callable
            function in ANY coordinates.
        """

    @property
    @abstractmethod
    def equilibrium_config(self) -> np.ndarray:
        """Equilibrium configuration
        NOTE: in a.u.

        Returns:
            self._equilibrium_config: (deg_of_freedom,) the configuration
                cartesian coordinates for each atom, flattened, in a.u..
                For example, CH4:
                self._equilibrium_config = np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        self._a_au,
                        self._a_au,
                        self._a_au,
                        -self._a_au,
                        -self._a_au,
                        self._a_au,
                        self._a_au,
                        -self._a_au,
                        -self._a_au,
                        -self._a_au,
                        self._a_au,
                        -self._a_au,
                    ],
                    dtype=np.float64,
                )
        """

    @property
    @abstractmethod
    def sqrt_atom_masses(self) -> np.ndarray:
        """Atom masses
        NOTE: in a.u.

        Returns:
            self._sqrt_atom_masses: (deg_of_greedom,) the atom masses mapping with
                the equilibrium_config. For example, CH4:
                self.sqrt_atom_masses = np.sqrt(
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

    @property
    @abstractmethod
    def _arg_sort_index(self) -> np.ndarray:
        """Reindexing Q arg sort index

        Returns:
            self.__arg_sort_index: (deg_of_freedom, ) the arg_sort result
                for reindexing Q.
                e.g. _arg_sort_index: from [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]:
                    the common notation of different harmonic frequencies,
                to [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6]
                    in the ascending order of harmonic frequencies,
                    s.t. w7 <= w8 <= w9 <= w2 <= w3 <= w1 <= w4 <= w5 <= w6.
        """

    @property
    @abstractmethod
    def _arg_sort_index_inverse(self) -> np.ndarray:
        """Inverse reindexing Q arg sort index

        Returns:
            self.__arg_sort_index_inverse: (deg_of_freedom, )
                the inverse arg_sort result
                for reindexing Q.
                e.g. _arg_sort_index_inverse:
                    from [0 0 0 0 0 0 w7 w8 w9 w2 w3 w1 w4 w5 w6]
                    in the ascending order of harmonic frequencies,
                    s.t. w7 <= w8 <= w9 <= w2 <= w3 <= w1 <= w4 <= w5 <= w6.
                to [0 0 0 0 0 0 w1 w2 w3 w4 w5 w6 w7 w8 w9]:
                    the common notation of different harmonic frequencies,
        """

    @property
    @abstractmethod
    def _zeros_in_redundant_normal(self) -> int:
        """Number of zeros to add in the redundant Q

        Returns:
            self.__zeros_in_redundant_normal: the zeros to append in redundant normal
                coordinates. This would be 6 for non-linear molecule and
                5 for linear molecule.
        """

    @property
    @abstractmethod
    def trans_normal2mwcd_matrix(self) -> np.ndarray:
        """The transformation matrix U from
        Redundant normal coordinates to
        mass-weighted cartesian displacement coordinates.
        NOTE: in a.u.

        Returns:
            self._trans_normal2mwcd_matrix: (deg_of_freedom,deg_of_freedom)
                the transformation matrix
                that transforms (redundant) normal coordinates to
                mass-weighted cartesian displacement coordinates.k.
        """

    @property
    @abstractmethod
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
    @abstractmethod
    def xalpha(self) -> float:
        """The scaled normal coordinates scaling factor.

        Returns:
            self._xalpha: the scaling factor of normal coordinates
                s.t. sqrt(1/alpha) x_i = Q_i, where x_i is the
                scaled_normal directly used in the programm
                and Q_i is the physical normal coordinates.
        """

    @property
    @abstractmethod
    def external_pes(self) -> bool:
        """Indication of the PES type.

        Returns:
            self._external_pes: if true, the corresponding
                coordinate transformation for PES would
                use jax functions, otherwise would use a
                pure numpy version.
        """

    def get_mass_weight(
        self,
        direction: str,
    ) -> np.ndarray:
        """Generate mass weight for mass-weighted cartesian
        displacement coordinates.
        NOTE: in a.u.

        Args:
            direction: denote which direction to perform conversion
                if "weight2config" then config_coor = coefficient * mass_weighted_coor
                if "config2weight" then mass_weighted_coor = coefficient * config_coor

        Returns:
            coefficient: (deg_of_freedom,) in the same order as flattened
                configuration cartesian displacement coordinates.
        """
        flatten_masses = self.sqrt_atom_masses
        if direction == "weight2config":
            coefficient = 1.0 / flatten_masses
        elif direction == "config2weight":
            coefficient = flatten_masses
        else:
            raise ValueError(f"Unexpected direction={direction}")
        return coefficient

    def ccd2mwcd(
        self,
        configuration_cartesian_displacement: jax.Array,
    ) -> jax.Array:
        """configuration cartesian displacement to mass-weight cartesian displacement

        Args:
            configuration_cartesian_displacement: (deg_of_freedom,) the flattened
                configureation cartesian displacement coordinates of molecule with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.

        Returns:
            mass_weight_cartesian_displacement: (deg_of_freedom,) the flattened mass
                weight cartesian displacement coordinates of molecule, with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.
        """
        coefficient = self.get_mass_weight(direction="config2weight")
        coefficient = jnp.array(coefficient)
        mass_weight_cartesian_displacement = (
            coefficient * configuration_cartesian_displacement
        )
        return mass_weight_cartesian_displacement

    def mwcd2ccd(
        self,
        mass_weight_cartesian_displacement: jax.Array,
    ) -> jax.Array:
        """mass-weight cartesian displacement to configuration cartesian displacement

        Args:
            mass_weight_cartesian_displacement: (deg_of_freedom,) the flattened mass
                weight cartesian displacement coordinates of molecule, with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.

        Returns:
            configuration_cartesian_displacement: (deg_of_freedom,) the flattened
                configureation cartesian displacement coordinates of CH4 with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.

        """
        coefficient = self.get_mass_weight(direction="weight2config")
        coefficient = jnp.array(coefficient)
        configuration_cartesian_displacement = (
            coefficient * mass_weight_cartesian_displacement
        )
        return configuration_cartesian_displacement

    def mwcd2ccd_np(
        self,
        mass_weight_cartesian_displacement: np.ndarray,
    ) -> np.ndarray:
        """mass-weight cartesian displacement to configuration cartesian displacement

        Args:
            mass_weight_cartesian_displacement: (deg_of_freedom,) the flattened mass
                weight cartesian displacement coordinates of molecule, with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.

        Returns:
            configuration_cartesian_displacement: (deg_of_freedom,) the flattened
                configureation cartesian displacement coordinates of CH4 with
                corresponding relationship stated in the docstring
                of pes_mwcd.
        NOTE: in a.u.

        """
        coefficient = self.get_mass_weight(direction="weight2config")
        configuration_cartesian_displacement = (
            coefficient * mass_weight_cartesian_displacement
        )
        return configuration_cartesian_displacement

    @abstractmethod
    def convert_mwcd_to_pes_origin_func_input(
        self,
        mass_weight_cartesian_displacement_a0: np.ndarray,
    ) -> np.ndarray:
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

    @abstractmethod
    def convert_config_cartesian_to_pes_origin_func_input(
        self,
        cartesian_coors: np.ndarray,
    ) -> np.ndarray:
        """Convert the configuration cartesian coordinates
        in a0 (atomic unit) into the input of the original PES function.
        NOTE: MUST be rewritten in subclass!

        Args:
            cartesian_coors: (dim * num_of_atoms,) the flattened
                cartesian coordinates in a.u.
                in a specific order. For example, for CH4:
            (C_x C_y C_z H_1x H_1y H_1z H_2x H_2y H_2z H_3x H_3y H_3z H_4x H_4y H_4z)
                NOTE: in atomic unit!

        Returns:
            pes_origin_func_input: the DIRECT original PES function input.
        """

    @abstractmethod
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

    def convert_normal_to_scaled_normal(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to scaled normal coordinates

        Args:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example,CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            scaled_normal: (num_of_modes,1) the scaled normal coordinates
                corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.
        """
        num_of_modes = normal_coor.shape[0]
        scaled_x_flattened = np.sqrt(self.xalpha) * normal_coor
        scaled_normal = scaled_x_flattened.reshape(num_of_modes, 1)
        return scaled_normal

    def convert_scaled_normal_to_normal(
        self,
        scaled_normal: jax.Array,
    ) -> jax.Array:
        """Convert scaled normal coordinates to normal coordinates

        Args:
            scaled_normal: (num_of_modes,1) the scaled normal coordinates
                corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.

        Returns:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        scaled_normal_flattened = jnp.hstack(scaled_normal)
        normal_coor = scaled_normal_flattened / np.sqrt(self.xalpha)
        return normal_coor

    def convert_scaled_normal_to_normal_np(
        self,
        scaled_normal: np.ndarray,
    ) -> np.ndarray:
        """Convert scaled normal coordinates to normal coordinates

        Args:
            scaled_normal: (num_of_modes,1) the scaled normal coordinates
                corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.

        Returns:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!
        """
        scaled_normal_flattened = scaled_normal.reshape(-1)
        normal_coor = scaled_normal_flattened / np.sqrt(self.xalpha)
        return normal_coor

    def convert_normal_to_mass_weight(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert normal coordinates to
        mass weighted cartesian displacement coordinates.

        Args:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            mass_weight_cartesian_displacement_a0: (deg_of_freedom,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the pes_mwcd.
                NOTE: in atomic unit!

        NOTE: Stop gradient to transform matrix here!
        """
        zeros = jnp.zeros(self._zeros_in_redundant_normal)
        normal_redundant = jnp.concatenate((zeros, normal_coor))
        normal_reindexed_redundant = normal_redundant[self._arg_sort_index]
        mass_weight_cartesian_displacement_a0 = jnp.einsum(
            "ij,j", self.trans_normal2mwcd_matrix, normal_reindexed_redundant
        )
        return mass_weight_cartesian_displacement_a0

    def convert_normal_to_mass_weight_np(
        self,
        normal_coor: np.ndarray,
    ) -> np.ndarray:
        """Convert normal coordinates to
        mass weighted cartesian displacement coordinates.

        Args:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            mass_weight_cartesian_displacement_a0: (deg_of_freedom,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the pes_mwcd.
                NOTE: in atomic unit!

        NOTE: Stop gradient to transform matrix here!
        """
        zeros = np.zeros(self._zeros_in_redundant_normal)
        normal_redundant = np.concatenate((zeros, normal_coor))
        normal_reindexed_redundant = normal_redundant[self._arg_sort_index]
        mass_weight_cartesian_displacement_a0 = np.einsum(
            "ij,j", np.array(self.trans_normal2mwcd_matrix), normal_reindexed_redundant
        )
        return mass_weight_cartesian_displacement_a0

    def convert_mass_weight_to_normal(
        self,
        mass_weight_cartesian_displacement_a0: jax.Array,
    ) -> jax.Array:
        """Convert mass weighted cartesian displacement coordinates to
        normal coordinates.

        Args:
            mass_weight_cartesian_displacement_a0: (deg_of_freedom,) the flattened
                mass weight cartesian displacement coordinates,
                order stated in the docstring of the pes_mwcd.
                NOTE: in atomic unit!

        Returns:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        NOTE: Stop gradient to transform matrix here!
        """
        trans_mwcd2normal_matrix = self.trans_normal2mwcd_matrix.T
        normal_reindexed_redundant = jnp.einsum(
            "ij,j", trans_mwcd2normal_matrix, mass_weight_cartesian_displacement_a0
        )
        normal_redundant = normal_reindexed_redundant[self._arg_sort_index_inverse]
        normal_coor = normal_redundant[6::]
        return normal_coor

    def normal_to_config_cartesian(
        self,
        normal_coor: jax.Array,
    ) -> jax.Array:
        """Convert the normal coordinates
        into configuration cartesian coordinates

        Args:
            normal_coor: (num_of_modes,1) the physical normal coordinates
                corresponding to
                the order of harmonic frequencies,
            NOTE: in a.u.

        Returns:
            configuration_cartesian_coordinates: (num_of_atoms,dim) the reshaped
                configuration cartesian coordinates of molecule. For example,
                if CH4:
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]
            NOTE: in a.u.
        """
        deg_of_freedom = self.equilibrium_config.shape[0]
        num_of_atmos = int(deg_of_freedom / 3)
        mass_weight = self.convert_normal_to_mass_weight(
            normal_coor=normal_coor,
        )
        config_cartesian_displacement = self.mwcd2ccd(
            mass_weight_cartesian_displacement=mass_weight,
        )
        configuration_cartesian_coordinates = (
            jnp.array(self.equilibrium_config) + config_cartesian_displacement
        ).reshape(num_of_atmos, 3)
        return configuration_cartesian_coordinates

    def config_cartesian_to_normal(
        self,
        configuration_cartesian_coordinates: jax.Array,
    ) -> jax.Array:
        """Convert configuration cartesian coordinates
        into the physical normal coordinates.

        Args:
            configuration_cartesian_coordinates: (num_of_atoms,dim) the reshaped
                configuration cartesian coordinates of molecule. For example,
                if CH4:
                [
                    [Cx, Cy, Cz,],
                    [H_1x, H_1y, H_1z,],
                    [H_2x, H_2y, H_2z,],
                    [H_3x, H_3y, H_3z,],
                    [H_4x, H_4y, H_4z,],
                ]
            NOTE: in a.u.

        Returns:
            normal_coor: (num_of_modes,1) the physical normal coordinates
                corresponding to
                the order of harmonic frequencies,
            NOTE: in a.u.
        """
        config_cartesian_coor_flattened = configuration_cartesian_coordinates.reshape(
            -1
        )
        config_cartesian_displacement = (
            config_cartesian_coor_flattened - self.equilibrium_config
        )
        mass_weight_a0 = self.ccd2mwcd(
            configuration_cartesian_displacement=config_cartesian_displacement
        )
        normal_coor = self.convert_mass_weight_to_normal(
            mass_weight_cartesian_displacement_a0=mass_weight_a0
        )
        num_of_modes = normal_coor.shape[0]
        normal_coor = normal_coor.reshape(num_of_modes, 1)
        return normal_coor

    def get_w_indices_from_pes_ad(
        self,
        pes: Callable,
        x_e: jax.Array,
        w_indices_rearrange_index: np.ndarray,
    ) -> np.ndarray:
        """Utilizing AD to compute w_indices directly from pes_origin_func!
        Get w_indices (sqrt eigvals excluded zeros) from
        neuralvib.potential.util.trans2normal._normal_to_cartesian_displacement
        NOTE: the inputs and outputs ALL in a.u.

        Args:
            pes: the function that take x as input, returns potential
                energy. Specificly this pes is the pes
                under mass-weighted cartesian displacement coordinates.
                NOTE: input and output of pes are all in a.u.
            x_e: (deg_of_freedom, ) the x at equilibrium configuration.
                Specifically x_e are all zeros since the pes ONLY
                recieves mass-weighted cartesian displacement coordinates
                and in this case the x at equilibrium configuration
                are all zeros.

        Returns:
            w_indices: (num_of_modes,) the harmonic frequencies omegas
                of the molecule. If degeneracy occurs,
                the ws are indexed sequentially. For example,
                if w1 is two-fold degenerated, then they are
                indexed as w1,w2.
        """
        _, eigval = normal_to_cartesian_displacement(pes=pes, x_e=x_e)
        eigval = jax.lax.stop_gradient(eigval)
        clip_start = self._zeros_in_redundant_normal
        w_indices = jnp.sqrt(eigval[clip_start::])
        # To make omegas from ascending order to the corresponding order
        # in ref, which is usually one particular order
        # defined in particular paper.
        w_indices = np.array(w_indices[w_indices_rearrange_index])
        return w_indices

    def get_transfer_matrix_from_pes_ad(
        self,
        pes: Callable,
        x_e: jax.Array,
    ) -> jax.Array:
        """Utilizing AD to compute transfer matrix directly from pes_origin_func!
        Same as
        neuralvib.potential.util.trans2normal._normal_to_cartesian_displacement
        NOTE: the inputs and outputs ALL in a.u.

        Args:
            pes: the function that take x as input, returns potential
                energy. Specificly this pes is the pes
                under mass-weighted cartesian displacement coordinates.
                NOTE: input and output of pes are all in a.u.
            x_e: (deg_of_freedom, ) the x at equilibrium configuration.
                Specifically x_e are all zeros since the pes ONLY
                recieves mass-weighted cartesian displacement coordinates
                and in this case the x at equilibrium configuration
                are all zeros.

        Returns:
            trans_normal2mwcd_matrix: (deg_of_freedom,deg_of_freedom) the
                transformation matrix
                that transforms (redundant) normal coordinates to
                mass-weighted cartesian displacement coordinates

        NOTE: eigval check should be performed within this method!
        """
        trans_normal2mwcd_matrix, _ = normal_to_cartesian_displacement(pes=pes, x_e=x_e)
        trans_normal2mwcd_matrix = jax.lax.stop_gradient(trans_normal2mwcd_matrix)
        return trans_normal2mwcd_matrix

    def pes_config_cartesian(
        self,
        cartesian_coors: np.ndarray,
    ) -> float:
        """the PES that called with configuration cartesian coordinates
        in a0 (atomic unit)
        coordinates (flattened) as input.

        Args:
            cartesian_coors: (dim * num_of_atoms,) the flattened
                mass weight cartesian displacement coordinates,
                in a specific order. For example, for CH4:
                (
                    C_x C_y C_z
                    H_1x H_1y H_1z
                    H_2x H_2y H_2z
                    H_3x H_3y H_3z
                    H_4x H_4y H_4z
                )
                NOTE: in atomic unit!

        Returns:
            potential_energy: corresponding potential energy in Hartree.
        """
        origin_pes_input = self.convert_config_cartesian_to_pes_origin_func_input(
            cartesian_coors=cartesian_coors,
        )
        potential_energy_origin = self.pes_origin_func(origin_pes_input)
        potential_energy_hartree = self.convert_pes_origin_func_energy_to_hartree(
            pes_origin_func_energy=potential_energy_origin
        )
        return potential_energy_hartree

    def pes_mwcd(
        self,
        mass_weight_cartesian_displacement_a0: np.ndarray,
    ) -> float:
        """the PES that called with mass weight cartesian displacements
        in a0 (atomic unit)
        coordinates (flattened) as input.

        Args:
            mass_weight_cartesian_displacement_a0: (dim * num_of_atoms,) the flattened
                mass weight cartesian displacement coordinates,
                in a specific order. For example, for CH4:
                (
                    C_x C_y C_z
                    H_1x H_1y H_1z
                    H_2x H_2y H_2z
                    H_3x H_3y H_3z
                    H_4x H_4y H_4z
                )
                NOTE: in atomic unit!

        Returns:
            potential_energy: corresponding potential energy in Hartree.
        """
        origin_pes_input = self.convert_mwcd_to_pes_origin_func_input(
            mass_weight_cartesian_displacement_a0=mass_weight_cartesian_displacement_a0
        )
        potential_energy_origin = self.pes_origin_func(origin_pes_input)
        potential_energy_hartree = self.convert_pes_origin_func_energy_to_hartree(
            pes_origin_func_energy=potential_energy_origin
        )
        return potential_energy_hartree

    def pes_normal_coor(self, normal_coor: np.ndarray) -> float:
        """Calculate potential energy from normal coordinates
        with input and output all in a.u.

        Args:
            normal_coor: (num_of_modes,) the normal coordinates corresponding to
                the order of harmonic frequencies, for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4]
            NOTE: in a.u.! that is, normal coordinates uses
            a0 (Bohr) as unit!

        Returns:
            potential_energy_wavenumber: the corresponding potential energy.
            NOTE: in a.u.! that is, in Eh (Hartree).
        """
        if self.external_pes:
            delta_x = self.convert_normal_to_mass_weight_np(
                normal_coor=normal_coor,
            )
        else:
            delta_x = self.convert_normal_to_mass_weight(
                normal_coor=normal_coor,
            )
        potential_energy = self.pes_mwcd(mass_weight_cartesian_displacement_a0=delta_x)
        return potential_energy

    def pes_scaled_normal(self, scaled_normal: np.ndarray) -> float:
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
            And Q_i is the ith normal coordinate following,
                for example, CH4:
                [w1 w2 w2 w3 w3 w3 w4 w4 w4] in a.u.

        Args:
            scaled_normal: (num_of_modes,1) the scaled normal coordinates
                corresponding to
                the order of harmonic frequencies, used in the main programm,
                in a.u.
        Returns:
            potential_energy_hartree: the corresponding potential energy.
            NOTE: in hartree, Not scaled!
        """
        if self.external_pes:
            normal_coor = self.convert_scaled_normal_to_normal_np(
                scaled_normal=scaled_normal
            )
        else:
            normal_coor = self.convert_scaled_normal_to_normal(
                scaled_normal=scaled_normal
            )
        potential_energy = self.pes_normal_coor(normal_coor=normal_coor)
        return potential_energy

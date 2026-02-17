"""Initialize Molecule"""

import heapq
import itertools
import re
from argparse import Namespace

import jax
import numpy as np

from neuralvib.molecule.molecule_base import MoleculeBase
from neuralvib.molecule.ch5plus.ch5_plus import CH5Plus
from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
    equilibrium_mccoy_jpca_2021_125_5849_5859,
)
from neuralvib.molecule.ch4.molecule import CH4
from neuralvib.utils.convert import (
    convert_hartree_to_inverse_cm,
    convert_inverse_cm_to_hartree,
)


class InitMolecule:
    """Initialize Molecule

    Attributes:
        self.w_indices: (num_of_modes,) the harmonic frequencies of the molecule. Often
            in a specific order, see the docstrings of corresponding
            molecule object for more details.
            NOTE: in a.u.
        self.pes_cartesian: the PES function that accept
            SINGLE config cartesian coordinates and return the SINGLE energy
                signature:
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
                returns:
                    potential_energy: corresponding potential energy in Hartree.
            NOTE: input and outputs ALL in scaled a.u.!
        self.equivariant_partitions: the partitions used in the
            equivariant flow, ONLY initialize if needed!
        self.molenet_molecule_obj: the molecule_obj used in MoleNetFlow,
            ONLY initialize if needed!
        self.pot_parallel_method: the str denoting which method to use
            when parallel potential energy function inside
            `EnergyEstimator`
        self.particles: the tuple with atoms names, for example,
                ("C","H","H","H","H","H")
        self.match_external: True for using external PES, indicating for
            not jitting on the workflow that contains PES.
        self.molecule: the molecule string.
    """

    def __init__(
        self,
        molecule: str,
        x_alpha: float,
        input_args: Namespace,
    ) -> None:
        """Init Molecule

        Args:
            molecule: the string denoting the molecule type.
            x_alpha: the scaling factor of normal coordinates
                s.t. sqrt(1/alpha) x_i = Q_i, where x_i is the
                scaled_normal directly used in the programm
                and Q_i is the physical normal coordinates.
            input_args: the command line input arguments.
        """
        select_potential = input_args.select_potential
        if molecule == "CH5+":
            assert input_args.num_of_particles == 6
            mole_instance = CH5Plus(select_potential=select_potential)
            mole_instance.xalpha = x_alpha
            w_indices = mole_instance.w_indices
            potential_func_cartesian = mole_instance.pes_config_cartesian
            # partitions in flow to be equivariant
            self.equivariant_partitions: list = [1]
            # the molecule_obj used in MoleNetFlow
            self.molenet_molecule_obj: MoleculeBase = mole_instance
        elif molecule == "CH4":
            assert input_args.num_of_particles == 5
            mole_instance = CH4(select_potential=select_potential)
            mole_instance.xalpha = x_alpha
            w_indices = mole_instance.w_indices
            potential_func_cartesian = mole_instance.pes_config_cartesian
            self.equivariant_partitions: list = [1]
            self.molenet_molecule_obj: MoleculeBase = mole_instance
        elif molecule == "User":
            # User specified potential
            raise NotImplementedError
        else:
            # Undefined mode, raise an error
            raise ValueError(
                f"Undifined behaviour in --molecule!\n"
                "For instruction, type 'python3 main.py --help'. \n"
                f"Currently get {molecule}"
            )

        match_external = re.search(r"External", select_potential, re.IGNORECASE)
        match_joblib = re.search(r"joblib", select_potential, re.IGNORECASE)
        if match_joblib:
            if not match_external:
                raise ValueError(
                    "When using joblib, must add `External` to the input"
                    " of select_potential, for example, `External.PySCF.DFT.Joblib`."
                    f"Get select_potential input `{select_potential}`"
                    "but key word `External` not found."
                )
            print("Using joblib parallel for pes...")
            self.pot_parallel_method = "joblib"
        else:
            if match_external:
                raise ValueError(
                    "When using jax vmap pes, must exclude `External` to the input"
                    " of select_potential, for example, `PySCF.DFT.pure`"
                    " or `PySCF.DFT.io`."
                    f"Get select_potential input `{select_potential}`"
                    "but with key word `External` offered."
                )
            print("Using jax.vmap for pes...")
            self.pot_parallel_method = "jax.vmap"

        self.match_external: bool = match_external
        self.molecule: str = molecule
        self.w_indices: np.ndarray | jax.Array = w_indices
        self.eq_config = mole_instance.equilibrium_config
        self.pes_cartesian = potential_func_cartesian
        self.particles = mole_instance.particles
        # particle mass in a.u.
        self.particle_mass = mole_instance.particle_mass
        self.mole_instance = mole_instance
        try:
            self.excite_gen_type = input_args.excite_gen_type
        except AttributeError:
            print("Excitation generation type not provided, defaulting to None.")
            self.excite_gen_type = None

    @staticmethod
    def generate_combinations(n_rows: int, n_cols: int) -> list:
        """Generate combinations for
        n_rows x n_cols matrix
        s.t. each row is unique and the sum
        of each row is in ascending order.

        For example, if n_rows = 12 and n_cols = 3
        Then the return is
            [[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 2],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1]]
        """
        sequence = []
        max_val = 0

        # Keep generating combinations until the desired number of rows is reached
        while len(sequence) < n_rows:
            # Generate all possible combinations for the current max_val
            # This creates combinations using numbers from 0 up to max_val
            possible_combinations = itertools.product(range(max_val + 1), repeat=n_cols)

            # Filter combinations to include only those where the maximum element
            # is exactly equal to the current max_val
            for combo in possible_combinations:
                if max(combo) == max_val:
                    if len(sequence) < n_rows:
                        sequence.append(list(combo))
                    else:
                        # Stop generating if n_rows is reached
                        break

            # Move to the next maximum value for the next layer of combinations
            max_val += 1

            if len(sequence) >= n_rows:
                break
        return sequence

    @staticmethod
    def generate_combinations_2(n_rows: int, n_cols: int) -> list:
        """Generate combinations for
        n_rows x n_cols matrix
        s.t. each row is unique and the
        columns are ordered as follow:
        first [n_cols*0],
        then C_{n_cols}^1, then C_{n_cols}^2
        Here C_n^m is the combination number of
        n and m.

        [(0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, 2),
            (0, 1, 1),
            (0, 2, 0),
            (1, 0, 1),
            (1, 1, 0),
            (2, 0, 0),
            (0, 1, 2),
            (0, 2, 1)]
        """
        max_val = 1
        while max_val**n_cols < n_rows:
            max_val += 1
        sequence = itertools.product(range(max_val), repeat=n_cols)
        sequence = sorted(sequence, key=sum)[:n_rows]
        return sequence

    def generate_combinations_3(self, num_orbs: int, n_cols: int) -> np.ndarray:
        """
        Generate excitation number configurations for a given number of orbitals,
        sorted by their total harmonic oscillator (HO) energy in ascending order.

        Args:
            num_orbs: Number of excitation configurations (rows) to generate.
            n_cols: Number of 1D oscillators (columns), typically num_particles * dim.

        Returns:
            sequence: (num_orbs, n_cols) array of excitation numbers, each row
                representing a unique configuration, sorted by increasing HO energy.
        """
        au2cm = convert_hartree_to_inverse_cm(1.0)
        particles = self.particles
        omega_for_wf_basis = self.omega_for_wf_basis
        num_of_particles = len(particles)
        dim = 3
        omegas = [[omega_for_wf_basis[particle]] * dim for particle in particles]
        omegas = np.array(omegas).flatten()
        assert len(omegas) == n_cols

        # Using min-heap to create a priority queue
        # (energy, combo_tuple)
        min_heap = [(0.0, tuple([0] * n_cols))]
        visited = {tuple([0] * n_cols)}
        sequence = []

        while len(sequence) < num_orbs and min_heap:
            energy, combo_tuple = heapq.heappop(min_heap)
            sequence.append(list(combo_tuple))

            for i in range(n_cols):
                new_combo = list(combo_tuple)
                new_combo[i] += 1
                new_combo_tuple = tuple(new_combo)

                if new_combo_tuple not in visited:
                    visited.add(new_combo_tuple)
                    new_energy = energy + omegas[i]
                    heapq.heappush(min_heap, (new_energy, new_combo_tuple))
        sequence = np.array(sequence, dtype=int)

        zpe = np.sum(omegas) * au2cm / 2
        truncate_energy = np.sum(sequence[-1] * omegas) * au2cm + zpe
        print(f"Basis index truncating at energy: {truncate_energy:.2f} cm-1")
        return sequence

    @staticmethod
    def duplication_check(arr: np.ndarray) -> None:
        """Check for duplication in the array"""
        assert len(arr.shape) >= 2, "1D array don't need to check duplication"
        unique_rows = np.unique(arr, axis=0)
        if len(unique_rows) != len(arr):
            raise ValueError("The array contain duplicates, which is unexpected.")

    @property
    def omega_for_wf_basis(self) -> dict:
        """Manually set omega for wf basis init
        From ZPE to calculate the omegas
        assume that each 1-d oscillators have the same omega
        NOTE: This should ONLY be used in
          initialization of HermiteFunction basis!
        NOTE: This is for init of real WF
        and is manually set as a flattener way
        which is to set m*w=sigma
        """
        cof = convert_inverse_cm_to_hartree(1.0)
        if self.molecule == "CH5+":
            # here sigma refers to the widening of wf
            # in unit of a.u.
            _sigma_C = 0.5
            _sigma_H = 0.5
            _omega_for_wf_basis = {
                "C": 1 / (_sigma_C**2 * self.particle_mass["C"]),
                "H": 1 / (_sigma_H**2 * self.particle_mass["H"]),
            }
        elif self.molecule == "CH4":
            _sigma_C = 0.5
            _sigma_H = 0.5
            _omega_for_wf_basis = {
                "C": 1 / (_sigma_C**2 * self.particle_mass["C"]),
                "H": 1 / (_sigma_H**2 * self.particle_mass["H"]),
            }
        print(f"Omega for wf basis: {_omega_for_wf_basis}")
        return _omega_for_wf_basis

    @property
    def omega_for_pretrain(self) -> dict:
        """Manually set omega for wf basis init
        NOTE: This is for use of pretrain target WF
        """
        if self.molecule == "CH5+":
            _sigma_C = 0.05
            _sigma_H = 0.10
            _omega_for_wf_basis = {
                "C": 1 / (_sigma_C**2 * self.particle_mass["C"]),
                "H": 1 / (_sigma_H**2 * self.particle_mass["H"]),
            }
        elif self.molecule == "CH4":
            _sigma_C = 0.1
            _sigma_H = 0.1
            _omega_for_wf_basis = {
                "C": 1 / (_sigma_C**2 * self.particle_mass["C"]),
                "H": 1 / (_sigma_H**2 * self.particle_mass["H"]),
            }
        else:
            raise NotImplementedError
        return _omega_for_wf_basis

    @property
    def pretrain_x0(self) -> np.ndarray:
        """The artificial configuration for pretraining

        Returns:
            _pretrain_x0: (num_of_particles,dim,) the configuration
                cartesian coordinates for each atom,  in a.u.
            For CH5+, specificlly,
                eq_conf = np.array(
                    [
                        [Cx, Cy, Cz],
                        [H1x, H1y, H1z],
                        [H2x, H2y, H2z],
                        [H3x, H3y, H3z],
                        [H4x, H4y, H4z],
                        [H5x, H5y, H5z],
                    ]
                )
                in which the coordinates are the atoms' manually set
                position, typically used for pretrain,
                setting carbon as the origin of the coordinate
                system.
                NOTE: the order of the hydrogen atoms are
                the same as in  J. Chem. Phys. 121, 4105-4116(2004),
                Brown, McCoy, Braams, Jin and Bowman.
        """
        if self.molecule == "CH5+":
            move_frac = 1.0
            _pretrain_x0 = self.eq_config
            _pretrain_x0 = _pretrain_x0.reshape(6, 3)
            _pretrain_x0 = _pretrain_x0 * move_frac
        elif self.molecule == "CH4":
            _pretrain_x0 = self.eq_config.reshape(5, 3)
        return _pretrain_x0

    def excitation_numbers(self, num_of_orb: int) -> np.ndarray:
        """Get the excitation numbers for the system

        Args:
            num_of_orb: the number of orbitals to calculate.

        Returns:
            excitation_numbers: (num_of_orb,num_of_particles*dim)
                the excitation number of each excitation state
                of each 1d-oscillator (of each 1d coordinate),
                in the same order as that in coors(flattened).
        """
        print(f"Excitation index generation type: {self.excite_gen_type}")
        if self.excite_gen_type == 1:
            _ex_gen_fun = self.generate_combinations
        elif self.excite_gen_type == 2:
            _ex_gen_fun = self.generate_combinations_2
        elif self.excite_gen_type == 3:
            _ex_gen_fun = self.generate_combinations_3
        else:
            raise ValueError(
                f"Excitation generation type {self.excite_gen_type} not supported."
            )
        if self.molecule == "CH5+":
            # assert (
            #     self.excite_gen_type == 3
            # ), "For CH5+ only excite_gen_type 3 is supported."
            n_rows = num_of_orb
            n_cols = 18
            sequence = _ex_gen_fun(n_rows, n_cols)
            excitation_numbers = np.array(sequence, dtype=int)
            # Print the full array
            np.set_printoptions(threshold=np.inf)
            print(f"Excitation numbers: {excitation_numbers}")
            # Restore default print options (optional)
            np.set_printoptions(threshold=1000)
            self.duplication_check(excitation_numbers)
            return excitation_numbers
        elif self.molecule == "CH4":
            n_rows = num_of_orb
            n_cols = len(self.particles) * 3
            sequence = _ex_gen_fun(n_rows, n_cols)
            excitation_numbers = np.array(sequence, dtype=int)
            np.set_printoptions(threshold=np.inf)
            print(f"Excitation numbers: {excitation_numbers}")
            np.set_printoptions(threshold=1000)
            self.duplication_check(excitation_numbers)
            return excitation_numbers
        else:
            raise NotImplementedError(
                f"Excitation numbers for {self.molecule} not implemented."
            )

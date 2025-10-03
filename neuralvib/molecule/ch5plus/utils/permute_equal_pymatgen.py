"""Full Permutative Equivalent Configs in pymatgen Molecule api"""

from itertools import permutations

from pymatgen.core import Molecule
import numpy as np

from neuralvib.molecule.ch5plus.equilibrium_config import (
    equilibrium_bowman_jpca_2006_110_1569_1574,
)
from neuralvib.molecule.ch5plus.stationary_points import (
    saddle_c2v_bowman_jpca_2006_110_1569_1574,
    saddle_cs_ii_bowman_jcp_121_4105_4116_2004,
)


class PymatgenFullPermutedConfigs:
    """Functionalities of full permuted configs"""

    def __init__(self) -> None:
        """Init"""
        self.equil_config = equilibrium_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        self.saddle_cs_ii = saddle_cs_ii_bowman_jcp_121_4105_4116_2004().reshape(6, 3)
        self.saddle_c2v = saddle_c2v_bowman_jpca_2006_110_1569_1574().reshape(6, 3)
        atoms = ["C", "H", "H", "H", "H", "H"]
        permute_index = [1, 2, 3, 4, 5]
        hydrogen_all_permute = np.array(list(permutations(permute_index)))
        self._full_global_minimums = {}
        self._full_cs_ii = {}
        self._full_c2v = {}
        self._full_match_dict = {}
        for i, hydrogen_permute in enumerate(hydrogen_all_permute, start=1):
            config_permutation = np.concatenate(
                (
                    np.array([0]),
                    hydrogen_permute,
                )
            )
            global_minimum_permuted = self.equil_config[config_permutation].tolist()
            cs_ii_permuted = self.saddle_cs_ii[config_permutation].tolist()
            c2v_permuted = self.saddle_c2v[config_permutation].tolist()
            minimum_molecule_permutated = Molecule(atoms, global_minimum_permuted)
            cs_ii_molecule_permuted = Molecule(atoms, cs_ii_permuted)
            c2v_molecule_permuted = Molecule(atoms, c2v_permuted)
            self._full_global_minimums[i] = minimum_molecule_permutated
            self._full_cs_ii[i] = cs_ii_molecule_permuted
            self._full_c2v[i] = c2v_molecule_permuted
            self._full_match_dict[i] = 0

    @property
    def full_global_minimums(self) -> dict:
        """The full permutative global minimums
        in pymatgen.core Molecule

        Returns:
            self._full_global_minimums: the dict
                with the order of the molecule as
                key, and the Molecule instance as value.
        """
        return self._full_global_minimums

    @property
    def full_cs_ii(self) -> dict:
        """The full permutative Cs(II) saddle points
        in pymatgen.core Molecule

        Returns:
            self._full_cs_ii: the dict
                with the order of the molecule as
                key, and the Molecule instance as value.
        """
        return self._full_cs_ii

    @property
    def full_c2v(self) -> dict:
        """The full permutative C2v saddle points
        in pymatgen.core Molecule

        Returns:
            self._full_c2v: the dict
                with the order of the molecule as
                key, and the Molecule instance as value.
        """
        return self._full_c2v

    @property
    def init_full_match_dict(self) -> dict:
        """The initialized full permutative
        matching dict

        Returns:
            self._full_match_dict: the dict with the
                order same as self.full_global_minimums,
                and values all initialized to zero for
                match count.
        """
        return self._full_match_dict

    @property
    def matcher_tolerance(self) -> float:
        """The matcher tolerance that seperates
        different permutatively equivalent global
        minimums.

        NOTE: since the default MoleculeMatcher takes
        identicle atoms as different ordered atoms,
        e.g. C H H H would be C H1 H2 H3,
        here matcher_tolerance is chosen such that
        each Cs(I) config with different hydrogen
        permutations are discernable.

        Returns:
            _matcher_tolerance: the matcher tolerance
                that seperates different permutatively
                equivalent Cs(I) configs.
        """
        _matcher_tolerance = 0.3
        return _matcher_tolerance

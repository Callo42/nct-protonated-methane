"""Stationary points config of CH5+"""

# %%
from numpy import dtype
import scipy
import numpy as np
import jax
import jax.numpy as jnp

from neuralvib.utils.convert import _convert_angstrom_to_a0
from neuralvib.utils.convert import _convert_a0_to_angstrom
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import (
    convert_cartesian_to_sorted_cm,
    load_flax_params,
)
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import get_nn_pes_input
from neuralvib.molecule.ch5plus.McCoy_NN_PES.trans2jax import McCoyNNPES

jax.config.update("jax_enable_x64", True)


def saddle_cs_ii_bowman_jcp_121_4105_4116_2004() -> np.ndarray:
    """Return Cs(II) saddle point configuration in cartesian
    coordinates from FIG 1 of J. Chem. Phys. 121, 4105-4116(2004),
    Brown, McCoy, Braams, Jin and Bowman.

    Returns:
        saddle_cs_ii: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, FLATTENED, in a.u.
            For CH5+, specificlly,
            saddle_cs_ii = np.array(
                [
                    Cx,Cy,Cz,
                    H1x,H1y,H1z,
                    H2x,H2y,H2z,
                    H3x,H3y,H3z,
                    H4x,H4y,H4z,
                    H5x,H5y,H5z,
                ]
            )
            setting carbon as the origin of the coordinate
            system.
            NOTE: the order of the hydrogen atoms are
            the same as in  J. Chem. Phys. 121, 4105-4116(2004),
            Brown, McCoy, Braams, Jin and Bowman.
    """
    saddle_cs_ii = np.array(
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            -4.19934200e-01,
            -9.10810017e-01,
            2.00216338e00,
            -4.19176684e-01,
            9.10913516e-01,
            2.00227507e00,
            -1.03723879e00,
            1.64228791e00,
            -7.23854751e-01,
            -1.03674874e00,
            -1.64206002e00,
            -7.25777286e-01,
            2.01823150e00,
            2.77223165e-04,
            -3.03365482e-01,
        ]
    )
    return saddle_cs_ii


def solve_saddle_cs_ii_bowman_jcp_121_4105_4116_2004() -> np.ndarray:
    """Return Cs(II) saddle point configuration in cartesian
    coordinates from FIG 1 of J. Chem. Phys. 121, 4105-4116(2004),
    Brown, McCoy, Braams, Jin and Bowman.

    Returns:
        saddle_cs_ii: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, FLATTENED, in a.u.
            For CH5+, specificlly,
            saddle_cs_ii = np.array(
                [
                    Cx,Cy,Cz,
                    H1x,H1y,H1z,
                    H2x,H2y,H2z,
                    H3x,H3y,H3z,
                    H4x,H4y,H4z,
                    H5x,H5y,H5z,
                ]
            )
            setting carbon as the origin of the coordinate
            system.
            NOTE: the order of the hydrogen atoms are
            the same as in  J. Chem. Phys. 121, 4105-4116(2004),
            Brown, McCoy, Braams, Jin and Bowman.
    """

    def equations(coors):
        """The constraints for solving equilibrium config
        xi,yi,zi denotes the cartesian configuration
        of each hydrogen atom, ordered as stated
        in the docstring above.
        NOTE: these bond lengths are directly retrieved
        from jcp 121, 4105(2004) and are in angstroms.
        """
        x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5 = coors
        d1 = x1**2 + y1**2 + z1**2 - 1.185**2
        d2 = x2**2 + y2**2 + z2**2 - 1.185**2
        d3 = x3**2 + y3**2 + z3**2 - 1.097**2
        d4 = x4**2 + y4**2 + z4**2 - 1.097**2
        d5 = x5**2 + y5**2 + z5**2 - 1.080**2
        r12 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2 - 0.964**2
        r15 = (x1 - x5) ** 2 + (y1 - y5) ** 2 + (z1 - z5) ** 2 - 1.840**2
        r23 = (x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2 - 1.529**2
        r24 = (x2 - x4) ** 2 + (y2 - y4) ** 2 + (z2 - z4) ** 2 - 2.004**2
        r34 = (x3 - x4) ** 2 + (y3 - y4) ** 2 + (z3 - z4) ** 2 - 1.738**2
        r35 = (x3 - x5) ** 2 + (y3 - y5) ** 2 + (z3 - z5) ** 2 - 1.849**2
        r45 = (x4 - x5) ** 2 + (y4 - y5) ** 2 + (z4 - z5) ** 2 - 1.849**2
        c1 = y5 - 0
        c2 = y1 + y2 - 0
        c3 = y3 + y4 - 0

        return [d1, d2, d3, d4, d5, r12, r15, r23, r24, r34, r35, r45, c1, c2, c3]

    # initial guess in a0 is from
    # Dyczmons, Staemmler and Kutzelnigg, Chem. Phys. Lett. 5, 361(1970)
    # Table 2, config IIIb, in a.u.
    # NOTE that the order in jcp 121, 4105(2004) is not the same as
    # in cpl 5, 361(1970). The corresponding relationships are:
    # assume Hi is the hydrogen atom in jcp and H'i is in cpl,
    #   H1 = H'4
    #   H2 = H'3
    #   H3 = H'2
    #   H4 = H'1
    #   H5 = H'5
    initial_guess_au = np.array(
        [
            -0.0738,
            0.9160,
            2.1174,
            -0.0738,
            -0.9160,
            2.1174,
            -0.9693,
            1.6781,
            -0.6340,
            -0.9693,
            -1.6781,
            -0.6340,
            1.9386,
            0.0,
            -0.6340,
        ]
    )
    init_guess_au_np = np.array(initial_guess_au)
    init_guess_ang = _convert_a0_to_angstrom(init_guess_au_np)
    # results.x in angstrom!
    solve_results = scipy.optimize.root(equations, init_guess_ang, method="lm")
    results_angstrom = solve_results.x
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5 = results_angstrom
    saddle_cs_ii_ang = np.array(
        [
            0,
            0,
            0,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            x3,
            y3,
            z3,
            x4,
            y4,
            z4,
            x5,
            y5,
            z5,
        ],
        dtype=np.float64,
    )
    saddle_cs_ii = _convert_angstrom_to_a0(saddle_cs_ii_ang)
    return saddle_cs_ii


def saddle_c2v_bowman_jpca_2006_110_1569_1574() -> np.ndarray:
    """Return  C2v saddle point configuration in cartesian
    coordinates from of J.Phys.Chem.A2006,110,1569-1574,
    which is the article of JBB Full PES

    Returns:
        saddle_c2v: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, FLATTENED, in a.u.
            For CH5+, specificlly,
            saddle_c2v = np.array(
                [
                    Cx,Cy,Cz,
                    H1x,H1y,H1z,
                    H2x,H2y,H2z,
                    H3x,H3y,H3z,
                    H4x,H4y,H4z,
                    H5x,H5y,H5z,
                ]
            )

    NOTE: This is a direct implement that utilizing
    JBBCH5ppotential.polar2cart(r).T
    and r is the C2v polar coordinates from ch5p.f90
    Then manually reordered s.t. the transfer matrix
    could be dirctly read from NN-PES one.
    """
    saddle_c2v = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [1.89754030e00, 0.00000000e00, 1.02745190e00],
            [1.10069284e-16, 1.79756783e00, -9.95301050e-01],
            [-3.30207853e-16, -1.79756783e00, -9.95301050e-01],
            [-1.89754030e00, 2.32381666e-16, 1.02745190e00],
            [0.00000000e00, 0.00000000e00, 2.19719000e00],
        ]
    )

    return saddle_c2v.reshape(-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    saddle_cs_ii = saddle_cs_ii_bowman_jcp_121_4105_4116_2004()
    saddle_cs_ii = saddle_cs_ii.reshape(6, 3)
    atoms_name = ("C", "H1", "H2", "H3", "H4", "H5")
    atoms = zip(atoms_name, saddle_cs_ii)
    atoms = dict(atoms)
    bonds = (
        ("H1", "C"),
        ("H2", "C"),
        ("H3", "C"),
        ("H4", "C"),
        ("H5", "C"),
    )

    # Create a new figure for plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    # Draw atoms
    for atom_name, position in atoms.items():
        if atom_name == "C":
            plot_color = "#9467BD"
            plot_size = 150
        else:
            plot_color = "#17BECF"
            plot_size = 50
        ax.scatter(
            position[0], position[1], position[2], color=plot_color, s=plot_size
        )  # s is the size of the sphere

    # Draw bonds
    for bond in bonds:
        atom1_pos = atoms[bond[0]]
        atom2_pos = atoms[bond[1]]
        ax.plot(
            (atom1_pos[0], atom2_pos[0]),
            (atom1_pos[1], atom2_pos[1]),
            (atom1_pos[2], atom2_pos[2]),
            color="k",
        )

    # Set plot labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Show the plot
    view_elev = 35
    view_azim = 230
    view_roll = None
    ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
    plt.show()

    print(f"saddle Cs(II) = \n{saddle_cs_ii}")

    # %%
    import sys

    sys.path.append("../../../")

# %%

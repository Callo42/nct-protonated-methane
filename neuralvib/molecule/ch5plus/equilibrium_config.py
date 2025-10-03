"""Equilibrium config of CH5+"""
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


def equilibrium_bowman_jpca_2006_110_1569_1574() -> np.ndarray:
    """Return equilibrium configuration in cartesian
    coordinates from of J.Phys.Chem.A2006,110,1569-1574,
    which is the article of JBB Full PES

    Returns:
        equil_conf_cartesian: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, FLATTENED, in a.u.
            For CH5+, specificlly,
            equil_conf_cartesian = np.array(
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

    NOTE: This is a direct implement that utilizing
    JBBCH5ppotential.polar2cart(r).T
    and r is the equilibrium polar coordinates from ch5p.f90
    Then manually reordered s.t. the transfer matrix
    could be dirctly read from NN-PES one.
    """
    equil_conf_cartesian = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [1.97750353, 0.0, -0.6904977],
            [-1.07107486, 1.43580626, -1.01070308],
            [-1.45117108, -1.70904469, -0.29321938],
            [0.17085674, -2.01988045, -1.00511179],
            [0.0, 0.0, 2.05676],
        ]
    )

    # reordered from above:
    # JBB Full PES | NN-pes
    #     3   |   1
    #     4   |   2
    #     1   |   3
    #     2   |   4
    #     5   |   5
    # equil_conf_cartesian = np.array(
    #     [
    #         [
    #             0.0,
    #             0.0,
    #             0.0,
    #         ],
    #         [-1.45117108, -1.70904469, -0.29321938],
    #         [0.17085674, -2.01988045, -1.00511179],
    #         [1.97750353, 0.0, -0.6904977],
    #         [-1.07107486, 1.43580626, -1.01070308],
    #         [0.0, 0.0, 2.05676],
    #     ]
    # )
    return equil_conf_cartesian.reshape(-1)


def equilibrium_bowman_jcp_121_4105_4116_2004() -> np.ndarray:
    """Return equilibrium configuration in cartesian
    coordinates from FIG 1 of J. Chem. Phys. 121, 4105-4116(2004),
    Brown, McCoy, Braams, Jin and Bowman.

    Returns:
        equil_conf_cartesian: (deg_of_freedom,) the configuration
            cartesian coordinates for each atom, FLATTENED, in a.u.
            For CH5+, specificlly,
            equil_conf_cartesian = np.array(
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

    def equations(coors):
        """The constraints for solving equilibrium config
        xi,yi,zi denotes the cartesian configuration
        of each hydrogen atom, ordered as stated
        in the docstring above.
        NOTE: these bond lengths are directly retrieved
        from jcp 121, 4105(2004) and are in angstroms.
        """
        x1, z1, x2, z2, x3, z3, x4, y4, z4, x5, y5, z5 = coors
        f1 = (x1 - x2) ** 2 + (z1 - z2) ** 2 - 0.976**2
        f2 = (x2 - x3) ** 2 + (z2 - z3) ** 2 - 1.406**2
        f3 = x1**2 + z1**2 - 1.180**2
        f4 = x2**2 + z2**2 - 1.183**2
        f5 = x3**2 + z3**2 - 1.107**2
        f6 = (x3 - x4) ** 2 + (y4) ** 2 + (z3 - z4) ** 2 - 1.779**2
        f7 = (x1 - x5) ** 2 + (y5) ** 2 + (z1 - z5) ** 2 - 1.711**2
        f8 = (x4 - x5) ** 2 + (y4 - y5) ** 2 + (z4 - z5) ** 2 - 1.877**2
        f9 = x4**2 + y4**2 + z4**2 - 1.085**2
        f10 = x5**2 + y5**2 + z5**2 - 1.085**2
        f11 = (x1 - x4) ** 2 + y4**2 + (z1 - z4) ** 2 - 1.711**2
        f12 = (x3 - x5) ** 2 + y5**2 + (z3 - z5) ** 2 - 1.779**2
        return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

    # initial guess in a0 is from
    # Dyczmons, Staemmler and Kutzelnigg, Chem. Phys. Lett. 5, 361(1970)
    # Table 2, config IIIa, in a.u.
    # NOTE that the order in jcp 121, 4105(2004) is not the same as
    # in cpl 5, 361(1970). The corresponding relationships are:
    # assume Hi is the hydrogen atom in jcp and H'i is in cpl,
    #   H1 = H'3
    #   H2 = H'4
    #   H3 = H'5
    #   H4 = H'1
    #   H5 = H'2
    # and here initial_guess_au omits y1,y2,y3 since they are all
    # equals to zero. (The H1, H2, H3 are located in the x-z plane)
    initial_guess_au = [
        -0.8234,
        2.1562,
        1.0079,
        2.0797,
        1.9386,
        -0.6340,
        -0.9693,
        -1.6781,
        -0.6340,
        -0.9693,
        1.6781,
        -0.6340,
    ]
    init_guess_au_np = np.array(initial_guess_au)
    init_guess_ang = _convert_a0_to_angstrom(init_guess_au_np)
    # results.x in angstrom!
    solve_results = scipy.optimize.root(equations, init_guess_ang, method="lm")
    results_angstrom = solve_results.x
    x1, z1, x2, z2, x3, z3, x4, y4, z4, x5, y5, z5 = results_angstrom
    equil_config_cartesian_angstrom = np.array(
        [
            0,
            0,
            0,
            x1,
            0,
            z1,
            x2,
            0,
            z2,
            x3,
            0,
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
    equil_config_cartesian = _convert_angstrom_to_a0(equil_config_cartesian_angstrom)
    return equil_config_cartesian


def from_bond_length_to_config(r1, r3, r4, r12, r15, r23, r34, r45) -> jnp.ndarray:
    """From given bond length, solve configuration.
    The fixed bond lengths in this function is
    chosen from Bowman's PES,
    J. Phys. Chem A 2006, 110, 1569-1574.
    NOTE: currently only for minimizing NN-PES!
    """

    def equations(coors):
        """The constraints for solving equilibrium config
        xi,yi,zi denotes the cartesian configuration
        of each hydrogen atom, ordered as stated
        in the docstring above.
        NOTE: these bond lengths are directly retrieved
        from jcp 121, 4105(2004) and are in angstroms.
        """
        x1, z1, x2, z2, x3, z3, x4, y4, z4, x5, y5, z5 = coors

        r2 = r1
        r5 = r4
        r14 = r15
        r35 = r34

        f3 = x1**2 + z1**2 - r1**2
        f4 = x2**2 + z2**2 - r2**2
        f5 = x3**2 + z3**2 - r3**2
        f9 = x4**2 + y4**2 + z4**2 - r4**2
        f10 = x5**2 + y5**2 + z5**2 - r5**2
        f1 = (x1 - x2) ** 2 + (z1 - z2) ** 2 - r12**2
        f2 = (x2 - x3) ** 2 + (z2 - z3) ** 2 - r23**2
        f8 = (x4 - x5) ** 2 + (y4 - y5) ** 2 + (z4 - z5) ** 2 - r45**2
        f6 = (x3 - x4) ** 2 + (y4) ** 2 + (z3 - z4) ** 2 - r34**2
        f12 = (x3 - x5) ** 2 + y5**2 + (z3 - z5) ** 2 - r35**2
        f7 = (x1 - x5) ** 2 + (y5) ** 2 + (z1 - z5) ** 2 - r15**2
        f11 = (x1 - x4) ** 2 + y4**2 + (z1 - z4) ** 2 - r14**2
        return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

    # initial guess in cartesian
    # _init_guess = np.array([
    #               -0.46197572,           2.18132244,
    #     1.34030521,           1.78911093,  1.93212932,          -0.80114603,
    #     -0.90638449, -1.77353116, -0.48546219, -0.90638449,  1.77353116, -0.48546219,
    # ])
    # results.x in angstrom!
    # solve_results = scipy.optimize.root(equations, _init_guess, method="lm")
    # results_angstrom = solve_results.x
    # x1, z1, x2, z2, x3, z3, x4, y4, z4, x5, y5, z5 = results_angstrom
    initial_guess_au = [
        -0.8234,
        2.1562,
        1.0079,
        2.0797,
        1.9386,
        -0.6340,
        -0.9693,
        -1.6781,
        -0.6340,
        -0.9693,
        1.6781,
        -0.6340,
    ]
    init_guess_au_np = jnp.array(initial_guess_au, dtype=jnp.float64)
    init_guess_ang = _convert_a0_to_angstrom(init_guess_au_np)
    # results.x in angstrom!
    solve_results = scipy.optimize.root(equations, init_guess_ang, method="lm")
    results_angstrom = solve_results.x
    x1, z1, x2, z2, x3, z3, x4, y4, z4, x5, y5, z5 = results_angstrom
    equil_config_cartesian_angstrom = jnp.array(
        [
            0,
            0,
            0,
            x1,
            0,
            z1,
            x2,
            0,
            z2,
            x3,
            0,
            z3,
            x4,
            y4,
            z4,
            x5,
            y5,
            z5,
        ],
        dtype=jnp.float64,
    )
    equil_config_cartesian = _convert_angstrom_to_a0(equil_config_cartesian_angstrom)
    return equil_config_cartesian


def equilibrium_mccoy_jpca_2021_125_5849_5859() -> np.ndarray:
    """Get equilibrium configuration by finding the
    minimun of NN-PES, and then convert it into
    cartesian configurations.

    NOTE: this is done by starting from the configuration
    of jcp 121, 4105-4116(2004) and then setting the five
    CH bonds and Hb-Hc bonds length as it is in Bowman's
    PES, then setting other bond lengths as searchable parameters,
    passing to scipy.optimize.minimize, then retrieving to
    the cartesian configuration.
    """

    equil_config = np.array(
        [
            0.0,
            0.0,
            0.0,
            -0.47973221,
            0.0,
            2.2111107,
            1.28476495,
            0.0,
            1.8620277,
            1.94164464,
            0.0,
            -0.78631647,
            -0.92408056,
            -1.77626236,
            -0.47195612,
            -0.92408056,
            1.77626236,
            -0.47195612,
        ],
        dtype=np.float64,
    )
    return equil_config


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def _plot(equil_config_ch5):

        equil_config_ch5 = equil_config_ch5.reshape(6, 3)
        atoms_name = ("C", "H1", "H2", "H3", "H4", "H5")
        atoms = zip(atoms_name, equil_config_ch5)
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
        view_elev = 25
        view_azim = 260
        view_roll = None
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
        plt.show()

    nn_pes_flax = McCoyNNPES(out_dims=1)
    params_flax_file = "./neuralvib/molecule/ch5plus/McCoy_NN_PES/params_flax"
    params_flax = load_flax_params(params_flax_file)

    r1 = 1.197
    r3 = 1.108
    r4 = 1.088
    r12 = 0.952
    r15 = 1.719
    r23 = 1.444
    r34 = 1.792
    r45 = 1.880

    equil_config_ch5 = from_bond_length_to_config(r1, r3, r4, r12, r15, r23, r34, r45)
    _plot(equil_config_ch5)

    input_to_nn = get_nn_pes_input(convert_cartesian_to_sorted_cm(equil_config_ch5))
    energy_nn_pes = nn_pes_flax.apply(params_flax, input_to_nn)
    print(f"Config energy by NN-PES:{energy_nn_pes}")

    # %%
    import scipy.optimize

    def pes_func(params):
        r1, r3, r4, r12, r15, r23, r34, r45 = params
        config = from_bond_length_to_config(r1, r3, r4, r12, r15, r23, r34, r45)
        input_to_nn = get_nn_pes_input(convert_cartesian_to_sorted_cm(config))
        return nn_pes_flax.apply(params_flax, input_to_nn)

    init_guess = np.array(
        [1.197, 1.108, 1.088, 0.952, 1.719, 1.444, 1.792, 1.880], dtype=np.float64
    )
    res = scipy.optimize.minimize(
        pes_func,
        init_guess,
    )
    print(res)

    # %%
    import os

    os.chdir("../../../")

# %%

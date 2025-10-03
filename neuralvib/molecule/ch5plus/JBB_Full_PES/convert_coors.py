"""Convert between coodinates"""

import numpy as np
import jax
import jax.numpy as jnp


def _calculate_angle_0_pi(vec_1: jax.Array, vec_2: jax.Array) -> jax.Array:
    """Calculate the angle between vec_1 and vec_2
    range [0,pi]

    Args:
        vec_1: (dim,) the coordinate of the first vector
        vec_2: (dim,) the coordinate of the second vector

    Returns:
        angle_rad: the angle between vec_1 and vec_2
    """
    dot_product = jnp.dot(vec_1, vec_2)
    cross_product = jnp.cross(vec_1, vec_2)
    x1 = jnp.linalg.norm(cross_product)
    y1 = dot_product

    angle_rad = jnp.arctan2(x1, y1)
    angle_rad = jnp.where(angle_rad >= 0, angle_rad, -angle_rad)

    return angle_rad


def _calculate_azimuthal(y, x) -> jax.Array:
    """Calculate the azitmuthal angle, range [0,2pi]

    Args:
        y: the vector projection on x-y plane, y coordinate
        x: the vector projection on x-y plane, x coordinate

    Returns:
        angle_rad: the azimuthal angle given (y,x,0)
    """
    angle_rad = jnp.arctan2(y, x)
    angle_rad = jnp.where(angle_rad >= 0, angle_rad, angle_rad + 2 * jnp.pi)

    return angle_rad


def _calculate_azimuthal_0_pi(y, x) -> jax.Array:
    """Calculate the azitmuthal angle, range [0,pi]

    Args:
        y: the vector projection on x-y plane, y coordinate
        x: the vector projection on x-y plane, x coordinate

    Returns:
        angle_rad: the azimuthal angle given (y,x,0)
    """
    angle_rad = jnp.arctan2(y, x)
    angle_rad = jnp.where(angle_rad >= 0, angle_rad, -angle_rad)

    return angle_rad


def config_cartesian2jbb_polar_input(
    cartesian_coors: jax.Array,
) -> jax.Array:
    """Convert configuration cartesian coordinates
    to polar coordinates for CH5+
    NOTE: for JBB Full PES input only!

    Args:
        cartesian_coors: (18,) the FLATTENED cartesian
            coordinates in a.u.

    Returns:
        polar_coors: (12,) the FLATTENED polar coordinates of CH5+:
        NOTE: units in a0 and radian!


        1  -  R1
        2  -  R2
        3  -  R3
        4  -  R4
        5  -  R5
        6  -  th1
        7  -  th2
        8  -  th3
        9  -  th4
        10  -  ph2, azimuthal angle of 2, 4-C-1 is x-z plane
        11  -  ph3, azimuthal angle of 3
        12  -  ph4, azimuthal angle of 4
    convention is pi >= ph2 >= 0, 2pi >= ph3,ph4 >=0
    """
    configuration_reshaped = cartesian_coors.reshape(6, 3)

    # NOTE: rearrange as stated in JBB PES's fortran file
    # ch5ppot.dist6.bond.f90
    coor_carbon = configuration_reshaped[0]
    coor_hydrogen_1 = configuration_reshaped[1]
    coor_hydrogen_2 = configuration_reshaped[2]
    coor_hydrogen_3 = configuration_reshaped[3]
    coor_hydrogen_4 = configuration_reshaped[4]
    coor_hydrogen_5 = configuration_reshaped[5]

    r1_vector = coor_hydrogen_1 - coor_carbon
    r2_vector = coor_hydrogen_2 - coor_carbon
    r3_vector = coor_hydrogen_3 - coor_carbon
    r4_vector = coor_hydrogen_4 - coor_carbon
    r5_vector = coor_hydrogen_5 - coor_carbon

    r1 = jnp.linalg.norm(r1_vector)
    r2 = jnp.linalg.norm(r2_vector)
    r3 = jnp.linalg.norm(r3_vector)
    r4 = jnp.linalg.norm(r4_vector)
    r5 = jnp.linalg.norm(r5_vector)

    zaxis_unit_vec = r5_vector / r5
    r5crossr1 = jnp.cross(r5_vector, r1_vector)
    yaxis_unit_vec = r5crossr1 / jnp.linalg.norm(r5crossr1)
    xaxis_unit_vec = jnp.cross(yaxis_unit_vec, zaxis_unit_vec)

    theta_15 = _calculate_angle_0_pi(r1_vector, r5_vector)
    theta_25 = _calculate_angle_0_pi(r2_vector, r5_vector)
    theta_35 = _calculate_angle_0_pi(r3_vector, r5_vector)
    theta_45 = _calculate_angle_0_pi(r4_vector, r5_vector)

    r2_zproject = jnp.dot(zaxis_unit_vec, r2_vector) * zaxis_unit_vec
    r2_xyproject = r2_vector - r2_zproject
    r2_xyproject_x = jnp.dot(xaxis_unit_vec, r2_xyproject)
    r2_xyproject_y = jnp.dot(yaxis_unit_vec, r2_xyproject)
    # ph_2 = _calculate_azimuthal_0_pi(r2_xyproject_y, r2_xyproject_x)
    ph_2 = _calculate_azimuthal(r2_xyproject_y, r2_xyproject_x)

    r3_zproject = jnp.dot(zaxis_unit_vec, r3_vector) * zaxis_unit_vec
    r3_xyproject = r3_vector - r3_zproject
    r3_xyproject_x = jnp.dot(xaxis_unit_vec, r3_xyproject)
    r3_xyproject_y = jnp.dot(yaxis_unit_vec, r3_xyproject)
    ph_3 = _calculate_azimuthal(r3_xyproject_y, r3_xyproject_x)

    r4_zproject = jnp.dot(zaxis_unit_vec, r4_vector) * zaxis_unit_vec
    r4_xyproject = r4_vector - r4_zproject
    r4_xyproject_x = jnp.dot(xaxis_unit_vec, r4_xyproject)
    r4_xyproject_y = jnp.dot(yaxis_unit_vec, r4_xyproject)
    ph_4 = _calculate_azimuthal(r4_xyproject_y, r4_xyproject_x)

    polar_coors = [
        r1,
        r2,
        r3,
        r4,
        r5,
        theta_15,
        theta_25,
        theta_35,
        theta_45,
        ph_2,
        ph_3,
        ph_4,
    ]

    return jnp.array(polar_coors, dtype=jnp.float64)


def config_cartesian2jbb_cartesian_input_xn(
    cartesian_coors: jax.Array | np.ndarray,
) -> jax.Array | np.ndarray:
    """Convert configuration cartesian coordinates
    to cartesian coordinates xn for CH5+
    NOTE: for JBB Full PES input only!

    Args:
        cartesian_coors: (18,) the FLATTENED cartesian
            coordinates in a.u.

    Returns:
        xn: (3,6)  is the Cartesian coordinates for
        six atoms in order of
        C H H H H H. (in bohr)
        with C as origin.
        NOTE:the recieved getpot recieves xn which is in shape(3,6)!
            see getpot.f for details.
    """
    cartesian_reshaped = cartesian_coors.reshape(6, 3)
    carbon = cartesian_reshaped[0]
    cartesian_reshaped_c_origin = cartesian_reshaped - carbon
    xn = cartesian_reshaped_c_origin.T
    return xn


def config_cartesian2jbb_ch5ppot_cart_input(
    cartesian_coors: jax.Array,
) -> jax.Array:
    """Convert configuration cartesian coordinates
    to cartesian coordinates of 5 H: cartr for CH5+
    NOTE: for JBB Full PES input only!
    NOTE: the direct input to ch5ppot_func_cart, cartr
        is the vector Cartesians setting carbon as
        origin.

    Args:
        cartesian_coors: (18,) the FLATTENED cartesian
            coordinates in a.u.

    Returns:
        cartr: (3,5)  is the Cartesian coordinates for
        FIVE atoms in order of
        H H H H H. (in bohr)
        NOTE:the recieved getpot recieves xn which is in shape(3,5)!
            see ch5ppot.dist6.bond.f90 for details.
    """
    cartesian_reshaped = cartesian_coors.reshape(6, 3)
    carbon = cartesian_reshaped[0]
    cartesian_reshaped_c_origin = cartesian_reshaped - carbon
    hydrogens = cartesian_reshaped_c_origin[1::]
    cartr = hydrogens.T
    return cartr

"""Network Utilities"""

import argparse

import jax
import haiku as hk

from neuralvib.molecule.utils.init_molecule import InitMolecule
from neuralvib.networks import flow_RNVP
from neuralvib.networks.flow_MoleNet import MoleNetFlow


def print_network_config(config: dict) -> None:
    """Pring the network settings

    Args:
        config: the dict that contains the network
            type and corresponding network configs.

    """
    raise NotImplementedError


def setting_network(
    input_args: argparse.Namespace,
) -> dict:
    """Reading from input args and return network config

    Args:
        input_args: the argparse.Namespace object returned
            from parser.parse_args()

    Returns:
        network_config: the network settings dict
            containing the network type, depth and
            relevant parameters.
            initialized as
            network_config = {
                "flow_type": flow_type,
                "flow_depth": flow_depth,
                "other_parameters": {},
                "flow_string": "Init",
            }
    """
    flow_type = input_args.flow_type
    flow_depth = input_args.flow_depth
    network_config = {
        "flow_type": flow_type,
        "flow_depth": flow_depth,
        "other_parameters": {},
        "flow_string": "Init",
    }
    if flow_type == "RNVP":
        # raise NotImplementedError(f"flow_type={flow_type} Not Implemented!")
        mlp_width, mlp_depth = input_args.mlp_width, input_args.mlp_depth
        network_config["other_parameters"] = {
            "mlp_width": mlp_width,
            "mlp_depth": mlp_depth,
        }
        flow_string = f"_rnvp_{flow_depth}_mlp_{mlp_width}_{mlp_depth}"
    elif flow_type == "FermiNetCH4":
        raise NotImplementedError(f"flow_type={flow_type} Not Implemented!")
        ferminet_spsize = input_args.ferminet_spsize
        ferminet_tpsize = input_args.ferminet_tpsize
        ferminet_init_stddev = input_args.ferminet_init_stddev
        network_config["other_parameters"] = {
            "ferminet_spsize": ferminet_spsize,
            "ferminet_tpsize": ferminet_tpsize,
            "ferminet_init_stddev": ferminet_init_stddev,
        }
        flow_string = (
            f"_ferminet_depth_{flow_depth}_spsize_{ferminet_spsize}"
            f"_tpsize_{ferminet_tpsize}"
        )
    elif flow_type == "MoleNet":
        molenet_spsize = input_args.molenet_spsize
        molenet_tpsize = input_args.molenet_tpsize
        molenet_init_stddev = input_args.molenet_init_stddev
        network_config["other_parameters"] = {
            "molenet_spsize": molenet_spsize,
            "molenet_tpsize": molenet_tpsize,
            "molenet_init_stddev": molenet_init_stddev,
        }
        flow_string = f"_molenet_{flow_depth}_{molenet_spsize}_{molenet_tpsize}"
    else:
        raise NotImplementedError(f"Currently not implemented for network={flow_type}")
    network_config["flow_string"] = flow_string

    return network_config


def make_flow(
    network_config: dict,
    key: jax.Array,
    input_args: argparse.Namespace,
    molecule_init_obj: InitMolecule,
) -> hk.Transformed:
    """Make hk.Transformed flow model

    Args:
        network_config: the dict containing network configs.
        key: the jax.random.PRNGKey array.
        input_args: the input arguments when invoking the program.
        molecule_init_obj: the molecule object instance initialized from
            InitMolecule.

    Returns:
        flow: the transformed network by haiku.transform.
    """
    if network_config["flow_type"] == "RNVP":
        # raise NotImplementedError(
        #     f"Network config = {network_config["flow_type"]} is not implemented!"
        # )
        flow = flow_RNVP.make_flow(
            key,
            network_config["flow_depth"],
            network_config["other_parameters"]["mlp_width"],
            network_config["other_parameters"]["mlp_depth"],
            input_args.num_of_particles * input_args.dim,
        )
    elif network_config["flow_type"] == "FermiNetCH4":
        raise NotImplementedError(
            f"Network config = {network_config["flow_type"]} is not implemented!"
        )
        # NOTE: currently onle implemented for CH4!
        # if input_args.molecule != "CH4":
        #     raise NotImplementedError(
        #         "Currently FermiNetCH4 is only implemented for CH4!\n"
        #     )
        # cartesian_coor_dim = 3
        # select_potential = "J.Chem.Phys.102,254-261(1995)"
        # ch4_object = CH4PESNormalCoor(select_potential=select_potential, alpha=1.0)

        # def flow_fn(x):
        #     flow = flow_FermiNet.FermiNetCH4(
        #         depth=network_config["flow_depth"],
        #         spsize=network_config["other_parameters"]["ferminet_spsize"],
        #         tpsize=network_config["other_parameters"]["ferminet_tpsize"],
        #         cartesian_coor_dim=cartesian_coor_dim,
        #         ch4_object=ch4_object,
        #         init_stddev=network_config["other_parameters"]["ferminet_init_stddev"],
        #     )
        #     return flow(x)

        # flow = hk.transform(flow_fn)
    elif network_config["flow_type"] == "MoleNet":

        def flow_fn(x):
            flow = MoleNetFlow(
                depth=network_config["flow_depth"],
                h1_size=network_config["other_parameters"]["molenet_spsize"],
                h2_size=network_config["other_parameters"]["molenet_tpsize"],
                partitions=molecule_init_obj.equivariant_partitions,
                init_stddev=network_config["other_parameters"]["molenet_init_stddev"],
            )
            return flow(x)

        flow = hk.transform(flow_fn)

    return flow

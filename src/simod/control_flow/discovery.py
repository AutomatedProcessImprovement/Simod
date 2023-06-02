import os
from pathlib import Path
from typing import List, Tuple

from simod.cli_formatter import print_step
from simod.control_flow.settings import HyperoptIterationParams
from simod.settings.control_flow_settings import (
    ProcessModelDiscoveryAlgorithm,
)
from simod.settings.simod_settings import PROJECT_DIR
from simod.utilities import is_windows, execute_external_command

sm2_path: Path = PROJECT_DIR / "external_tools/splitminer2/sm2.jar"
sm3_path: Path = PROJECT_DIR / "external_tools/splitminer3/bpmtk.jar"


def discover_process_model(log_path: Path, output_model_path: Path, params: HyperoptIterationParams):
    """
    Run the process model discovery algorithm specified in the [params] to discover
    a process model in [output_model_path] from the (XES) event log in [log_path].

    :param log_path: Path to the event log in XES format for the Split Miner algorithms.
    :param output_model_path: Path to write the discovered process model.
    :param params: configuration class specifying the process model discovery algorithm and its parameters.
    """
    if params.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
        # Run Split Miner 2
        discover_process_model_with_split_miner_2(log_path, output_model_path, params.concurrency)
    elif params.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
        # Run Split Miner 3
        discover_process_model_with_split_miner_3(
            log_path,
            output_model_path,
            params.eta,
            params.epsilon,
            params.prioritize_parallelism,
            params.replace_or_joins,
        )
    else:
        raise ValueError(f"Unknown process model discovery algorithm: {params.mining_algorithm}")
    # Assert that the process model was discovered
    assert output_model_path.exists(), f"Error trying to discover the process model in '{output_model_path}'."


def discover_process_model_with_split_miner_2(log_path: Path, output_model_path: Path, concurrency: float):
    """
    Discover, with Split Miner 2, a process model using the (XES) log in [log_path].

    :param log_path: Path to the event log in XES format.
    :param output_model_path: Path to write the discovered process model.
    :param concurrency: concurrency threshold.
    """
    # Define args depending on the system is running
    args, split_miner_path, input_log_path, model_output_path = _prepare_split_miner_params(
        sm2_path, log_path, output_model_path
    )
    # Prepare command structure
    args += [
        "-cp",
        split_miner_path,
        "au.edu.unimelb.services.ServiceProvider",
        "SM2",
        input_log_path,
        model_output_path,
        str(concurrency),
    ]
    # Execute command
    print_step(f"SplitMiner2 is running with the following arguments: {args}")
    execute_external_command(args)


def discover_process_model_with_split_miner_3(
    log_path: Path,
    output_model_path: Path,
    eta: float,
    epsilon: float,
    prioritize_parallelism: bool,
    replace_or_joins: bool,
):
    """
    Discover, with Split Miner 3, a process model using the (XES) log in [log_path].

    :param log_path: Path to the event log in XES format.
    :param output_model_path: Path to write the discovered process model.
    :param eta: percentile for frequency threshold.
    :param epsilon: parallelism threshold.
    :param prioritize_parallelism: boolean flag denoting if SM3 should prioritize parallelism over loops.
    :param replace_or_joins: boolean flag denoting if SM3 should replace non-trivial OR joins.
    """
    # Define args depending on the system is running
    args, split_miner_path, input_log_path, model_output_path = _prepare_split_miner_params(
        sm3_path, log_path, output_model_path
    )
    # Prepare command structure
    args += [
        "-cp",
        split_miner_path,
        "au.edu.unimelb.services.ServiceProvider",
        "SMD",
        str(eta),
        str(epsilon),
        str(prioritize_parallelism).lower(),  # Prioritize parallelism over loops
        str(replace_or_joins).lower(),  # Replace non-trivial OR joins
        "false",  # Remove loop activity markers (false increases model complexity)
        input_log_path,
        model_output_path,
    ]
    # Execute command
    print_step(f"SplitMiner3 is running with the following arguments: {args}")
    execute_external_command(args)


def _prepare_split_miner_params(
    split_miner: Path, log_path: Path, output_model_path: Path
) -> Tuple[List[str], str, str, str]:
    if is_windows():
        # Windows: ';' as separator and escape string with '"'
        args = ["java"]
        split_miner_path = '"' + str(split_miner) + ";" + os.path.join(os.path.dirname(split_miner), "lib", "*") + '"'
        input_log_path = '"' + str(log_path) + '"'
        model_output_path = '"' + str(output_model_path.with_suffix("")) + '"'
    else:
        # Linux: ':' as separator and add memory specs
        args = ["java", "-Xmx2G", "-Xms1024M"]
        split_miner_path = str(split_miner) + ":" + os.path.join(os.path.dirname(split_miner), "lib", "*")
        input_log_path = str(log_path)
        model_output_path = str(output_model_path.with_suffix(""))
    # Return params
    return args, split_miner_path, input_log_path, model_output_path

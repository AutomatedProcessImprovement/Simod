from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from simod.cli_formatter import print_step
from simod.control_flow.settings import HyperoptIterationParams
from simod.settings.control_flow_settings import (
    ProcessModelDiscoveryAlgorithm,
)
from simod.utilities import execute_external_command, is_windows

split_miner_jar_path: Path = Path(__file__).parent / "lib/splitminer-1.6-all.jar"


def discover_process_model(log_path: Path, output_model_path: Path, params: HyperoptIterationParams):
    """
    Run the process model discovery algorithm specified in the [params] to discover
    a process model in [output_model_path] from the (XES) event log in [log_path].

    :param log_path: Path to the event log in XES format for the Split Miner algorithms.
    :param output_model_path: Path to write the discovered process model.
    :param params: configuration class specifying the process model discovery algorithm and its parameters.
    """
    if params.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V1:
        discover_process_model_with_split_miner_v1(
            SplitMinerV1Settings(
                log_path,
                output_model_path,
                params.eta,
                params.epsilon,
                params.prioritize_parallelism,
                params.replace_or_joins,
            )
        )
    elif params.mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_V2:
        discover_process_model_with_split_miner_v2(SplitMinerV2Settings(log_path, output_model_path, params.epsilon))
    else:
        raise ValueError(f"Unknown process model discovery algorithm: {params.mining_algorithm}")

    assert output_model_path.exists(), f"Error trying to discover the process model in '{output_model_path}'."


@dataclass
class SplitMinerV1Settings:
    log_path: Path
    output_model_path: Path
    eta: float
    epsilon: float
    parallelism_first: bool  # Prioritize parallelism over loops
    replace_or_joins: bool  # Replace non-trivial OR joins
    remove_loop_activity_markers: bool = False  # False increases model complexity


@dataclass
class SplitMinerV2Settings:
    """
    Original author of Split Miner hardcoded eta, parallelism_first, replace_or_joins, and remove_loop_activity_markers
    values into the algorithm. It might have been done because it gives better results, but it is not clear.
    We pass only epsilon to Split Miner 2 for now.
    """

    log_path: Path
    output_model_path: Path
    epsilon: float


def discover_process_model_with_split_miner_v1(settings: SplitMinerV1Settings):
    global split_miner_jar_path

    args, split_miner_path, input_log_path, model_output_path = _prepare_split_miner_params(
        split_miner_jar_path, settings.log_path, settings.output_model_path, strip_output_suffix=False
    )

    args += [
        "-jar",
        split_miner_path,
        "--logPath",
        input_log_path,
        "--outputPath",
        model_output_path,
        "--eta",
        str(settings.eta),
        "--epsilon",
        str(settings.epsilon),
    ]

    # Boolean flags added only when they are True
    if settings.parallelism_first:
        args += ["--parallelismFirst"]
    if settings.replace_or_joins:
        args += ["--replaceIORs"]
    if settings.remove_loop_activity_markers:
        args += ["--removeLoopActivityMarkers"]

    print_step(f"SplitMiner v1 is running with the following arguments: {args}")
    execute_external_command(args)


def discover_process_model_with_split_miner_v2(settings: SplitMinerV2Settings):
    global split_miner_jar_path

    assert settings.epsilon is not None, "Epsilon must be provided for Split Miner v2."

    args, split_miner_path, input_log_path, model_output_path = _prepare_split_miner_params(
        split_miner_jar_path, settings.log_path, settings.output_model_path, strip_output_suffix=False
    )

    args += [
        "-jar",
        split_miner_path,
        "--logPath",
        input_log_path,
        "--outputPath",
        model_output_path,
        "--epsilon",
        str(settings.epsilon),
        "--splitminer2",  # Boolean flag is always added here to run Split Miner v2
    ]

    print_step(f"SplitMiner v2 is running with the following arguments: {args}")
    execute_external_command(args)


def _prepare_split_miner_params(
    split_miner: Path,
    log_path: Path,
    output_model_path: Path,
    strip_output_suffix: bool = True,
    headless: bool = True,
) -> Tuple[List[str], str, str, str]:
    if is_windows():
        # Windows: ';' as separator and escape string with '"'
        args = ["java"]
        if headless:
            args += ["-Djava.awt.headless=true"]
        split_miner_path = '"' + str(split_miner) + '"'
        input_log_path = '"' + str(log_path) + '"'
        if strip_output_suffix:
            model_output_path = '"' + str(output_model_path.with_suffix("")) + '"'
        else:
            if ".bpmn" not in str(output_model_path):
                model_output_path = str(output_model_path.with_suffix(".bpmn"))
            else:
                model_output_path = '"' + str(output_model_path) + '"'
    else:
        # Linux: ':' as separator and add memory specs
        args = ["java", "-Xmx2G", "-Xms1024M"]
        if headless:
            args += ["-Djava.awt.headless=true"]
        split_miner_path = str(split_miner)
        input_log_path = str(log_path)
        if strip_output_suffix:
            model_output_path = str(output_model_path.with_suffix(""))
        else:
            if ".bpmn" not in str(output_model_path):
                model_output_path = str(output_model_path.with_suffix(".bpmn"))
            else:
                model_output_path = str(output_model_path)

    return args, split_miner_path, input_log_path, model_output_path

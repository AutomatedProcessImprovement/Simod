import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs, read_csv_log
from prosimos.simulation_engine import run_simulation

from simod.cli_formatter import print_message, print_notice, print_warning
from simod.metrics import compute_metric
from ..settings.common_settings import Metric

cpu_count = multiprocessing.cpu_count()


@dataclass
class ProsimosSettings:
    """
    Configuration settings for running a Prosimos simulation.

    Attributes
    ----------
    bpmn_path : :class:`pathlib.Path`
        Path to the BPMN process model.
    parameters_path : :class:`pathlib.Path`
        Path to the Prosimos simulation parameters JSON file.
    output_log_path : :class:`pathlib.Path`
        Path to store the generated simulation log.
    num_simulation_cases : int
        Number of cases to simulate.
    simulation_start : :class:`pandas.Timestamp`
        Start timestamp for the simulation.
    """

    bpmn_path: Path
    parameters_path: Path
    output_log_path: Path
    num_simulation_cases: int
    simulation_start: pd.Timestamp


def simulate(settings: ProsimosSettings):
    """
    Runs a Prosimos simulation with the provided settings.

    Parameters
    ----------
    settings : :class:`ProsimosSettings`
        Configuration settings containing paths and parameters for the simulation.

    Notes
    -----
    - The function prints the simulation settings and invokes `run_simulation()`.
    - The labels of the start event, end event, and event timers are**not** recorded to the output log.
    - The simulation generates a process log stored in `settings.output_log_path`.
    """
    print_message(f"Simulation settings: {settings}")

    run_simulation(
        bpmn_path=settings.bpmn_path.__str__(),
        json_path=settings.parameters_path.__str__(),
        total_cases=settings.num_simulation_cases,
        stat_out_path=None,  # No statistics
        log_out_path=settings.output_log_path.__str__(),
        starting_at=settings.simulation_start.isoformat(),
        is_event_added_to_log=False,  # Don't add Events (start/end/timers) to output log
    )


def simulate_and_evaluate(
    process_model_path: Path,
    parameters_path: Path,
    output_dir: Path,
    simulation_cases: int,
    simulation_start_time: pd.Timestamp,
    validation_log: pd.DataFrame,
    validation_log_ids: EventLogIDs,
    metrics: List[Metric],
    num_simulations: int = 1,
) -> List[dict]:
    """
    Simulates a process model using Prosimos multiple times and evaluates the results.

    This function runs the simulation `num_simulations` times in parallel,
    compares the generated logs with a validation log, and evaluates them using provided metrics.

    Parameters
    ----------
    process_model_path : :class:`pathlib.Path`
        Path to the BPMN process model.
    parameters_path : :class:`pathlib.Path`
        Path to the Prosimos simulation parameters JSON file.
    output_dir : :class:`pathlib.Path`
        Directory where simulated logs will be stored.
    simulation_cases : int
        Number of cases to simulate per run.
    simulation_start_time : :class:`pandas.Timestamp`
        Start timestamp for the simulation.
    validation_log : :class:`pandas.DataFrame`
        The actual event log to compare against.
    validation_log_ids : :class:`EventLogIDs`
        Column mappings for identifying events in the validation log.
    metrics : List[:class:`~simod.settings.common_settings.Metric`]
        A list of metrics used to evaluate the simulated logs.
    num_simulations : int, optional
        Number of parallel simulation runs (default is 1).

    Returns
    -------
    List[dict]
        A list of evaluation results, one for each simulated log.

    Notes
    -----
    - Uses multiprocessing to speed up simulation when `num_simulations > 1`.
    - Simulated logs are automatically compared with `validation_log`.
    """

    simulation_log_paths = simulate_in_parallel(
        process_model_path, num_simulations, output_dir, parameters_path, simulation_cases, simulation_start_time
    )

    evaluation_measurements = evaluate_logs(metrics, simulation_log_paths, validation_log, validation_log_ids)

    return evaluation_measurements


def simulate_in_parallel(
    process_model_path: Path,
    num_simulations: int,
    output_dir: Path,
    parameters_path: Path,
    simulation_cases: int,
    simulation_start_time: pd.Timestamp,
) -> List[Path]:
    """
    Simulates a process model using Prosimos num_simulations times in parallel.

    :param process_model_path: Path to the BPMN model.
    :param num_simulations: Number of simulations to run in parallel. Default: 1. Each simulation produces a log.
    :param output_dir: Path to the output directory for simulated logs.
    :param parameters_path: Path to the Prosimos parameters.
    :param simulation_cases: Number of cases to simulate.
    :param simulation_start_time: Start time of the simulation.
    :return: Paths to the simulated logs.
    """
    global cpu_count

    w_count = min(num_simulations, cpu_count)

    simulation_arguments = [
        ProsimosSettings(
            bpmn_path=process_model_path,
            parameters_path=parameters_path,
            output_log_path=output_dir / f"simulated_log_{rep}.csv",
            num_simulation_cases=simulation_cases,
            simulation_start=simulation_start_time,
        )
        for rep in range(num_simulations)
    ]

    print_notice(f"Simulating {len(simulation_arguments)} times with {w_count} workers")

    with Pool(w_count) as pool:
        pool.map(simulate, simulation_arguments)

    simulation_log_paths = [simulation_argument.output_log_path for simulation_argument in simulation_arguments]

    return simulation_log_paths


def evaluate_logs(
    metrics: List[Metric],
    simulation_log_paths: List[Path],
    validation_log: pd.DataFrame,
    validation_log_ids: EventLogIDs,
) -> List[dict]:
    """
    Calculates the evaluation metrics for the simulated logs comparing it with the validation log.
    """
    global cpu_count

    w_count = min(len(simulation_log_paths), cpu_count)

    # Read simulated logs

    read_arguments = [
        (simulation_log_paths[index], PROSIMOS_LOG_IDS, index) for index in range(len(simulation_log_paths))
    ]

    print_notice(f"Reading {len(read_arguments)} simulated logs with {w_count} workers")

    with Pool(w_count) as pool:
        simulated_logs = pool.map(_read_simulated_log, read_arguments)

    # Evaluate

    evaluation_arguments = [
        (validation_log, validation_log_ids, log, PROSIMOS_LOG_IDS, metrics) for log in simulated_logs
    ]

    print_notice(f"Evaluating {len(evaluation_arguments)} simulated logs with {w_count} workers")

    with Pool(w_count) as pool:
        evaluation_measurements = pool.map(_evaluate_logs_using_metrics, evaluation_arguments)
    evaluation_measurements = list(itertools.chain.from_iterable(evaluation_measurements))

    return evaluation_measurements


def _read_simulated_log(arguments: Tuple):
    log_path, log_ids, simulation_repetition_index = arguments

    df = read_csv_log(log_path, log_ids=log_ids)

    df["role"] = df["resource"]
    df["source"] = "simulation"
    df["run_num"] = simulation_repetition_index

    return df


def _evaluate_logs_using_metrics(arguments: Tuple) -> List[dict]:
    validation_log: pd.DataFrame = arguments[0]
    validation_log_ids: EventLogIDs = arguments[1]
    simulated_log: pd.DataFrame = arguments[2]
    simulated_log_ids: EventLogIDs = arguments[3]
    metrics: List[Metric] = arguments[4]

    if len(simulated_log) > 0:
        rep = simulated_log.iloc[0].run_num
    else:
        print_warning("Error with the simulation! Trying to evaluate an empty simulated log.")
        rep = -1

    measurements = []
    for metric in metrics:
        value = compute_metric(metric, validation_log, validation_log_ids, simulated_log, simulated_log_ids)
        measurements.append({"run_num": rep, "metric": metric, "distance": value})

    return measurements

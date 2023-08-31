import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pix_framework.discovery.gateway_probabilities import GatewayProbabilities
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.discovery.resource_calendar_and_performance.resource_activity_performance import (
    ActivityResourceDistribution,
)
from pix_framework.discovery.resource_profiles import ResourceProfile
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs, read_csv_log
from prosimos.simulation_engine import run_simulation

from simod.cli_formatter import print_message, print_notice, print_warning
from simod.metrics import compute_metric
from ..settings.common_settings import Metric

cpu_count = multiprocessing.cpu_count()


@dataclass
class SimulationParameters:
    """
    Prosimos simulation parameters.
    """

    resource_profiles: List[ResourceProfile]
    resource_calendars: Dict[str, RCalendar]
    task_resource_distributions: List[ActivityResourceDistribution]
    arrival_distribution: dict
    arrival_calendar: RCalendar
    gateway_branching_probabilities: List[GatewayProbabilities]
    event_distribution: Optional[dict]

    def to_dict(self) -> dict:
        """Dictionary compatible with Prosimos."""
        parameters = {
            "resource_profiles": [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            "resource_calendars": [
                {
                    "id": self.resource_calendars[calendar_id].calendar_id,
                    "name": self.resource_calendars[calendar_id].calendar_id,
                    "time_periods": self.resource_calendars[calendar_id].to_json(),
                }
                for calendar_id in self.resource_calendars
            ],
            "task_resource_distribution": [
                activity_resources.to_dict() for activity_resources in self.task_resource_distributions
            ],
            "arrival_time_distribution": self.arrival_distribution,
            "arrival_time_calendar": self.arrival_calendar.to_json(),
            "gateway_branching_probabilities": [
                gateway_probabilities.to_dict() for gateway_probabilities in self.gateway_branching_probabilities
            ],
        }

        if self.event_distribution:
            parameters["event_distribution"] = self.event_distribution

        return parameters

    def to_json_file(self, file_path: Path) -> None:
        """JSON compatible with Prosimos."""
        with file_path.open("w") as f:
            json.dump(self.to_dict(), f)


@dataclass
class ProsimosSettings:
    """Prosimos simulator settings."""

    bpmn_path: Path
    parameters_path: Path
    output_log_path: Path
    num_simulation_cases: int
    simulation_start: pd.Timestamp


def simulate(settings: ProsimosSettings):
    """
    Simulates a process model using Prosimos.
    :param settings: Prosimos settings.
    :return: None.
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
    Simulates a process model using Prosimos num_simulations times in parallel.

    :param process_model_path: Path to the BPMN model.
    :param parameters_path: Path to the Prosimos parameters.
    :param output_dir: Path to the output directory for simulated logs.
    :param simulation_cases: Number of cases to simulate.
    :param simulation_start_time: Start time of the simulation.
    :param validation_log: Validation log.
    :param validation_log_ids: Validation log IDs.
    :param metrics: Metrics to evaluate the simulated logs with.
    :param num_simulations: Number of simulations to run in parallel. Default: 1. More simulations increase
        the accuracy of evaluation metrics.
    :return: Evaluation metrics.
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

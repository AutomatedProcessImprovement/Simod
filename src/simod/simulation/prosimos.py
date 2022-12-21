import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union

import pandas as pd

from simod.cli_formatter import print_notice, print_step
from simod.utilities import execute_shell_cmd
from .parameters.activity_resources import ActivityResourceDistribution
from .parameters.calendars import Calendar
from .parameters.gateway_probabilities import GatewayProbabilities
from .parameters.resource_profiles import ResourceProfile
from ..configuration import Metric
from ..event_log.column_mapping import PROSIMOS_COLUMNS, EventLogIDs
from ..event_log.utilities import read
from ..metrics.metrics import compute_metric

cpu_count = multiprocessing.cpu_count()


@dataclass
class SimulationParameters:
    """
    Prosimos simulation parameters.
    """

    resource_profiles: List[ResourceProfile]
    resource_calendars: List[Calendar]
    task_resource_distributions: List[ActivityResourceDistribution]
    arrival_distribution: dict
    arrival_calendar: Calendar
    gateway_branching_probabilities: Union[List[GatewayProbabilities], List[dict]]
    event_distribution: Optional[dict]

    def to_dict(self) -> dict:
        """Dictionary compatible with Prosimos."""
        parameters = {
            'resource_profiles':
                [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            'resource_calendars':
                [calendar.to_dict() for calendar in self.resource_calendars],
            'task_resource_distribution':
                [activity_resources.to_dict() for activity_resources in self.task_resource_distributions],
            'arrival_time_distribution':
                self.arrival_distribution,
            'arrival_time_calendar':
                self.arrival_calendar.to_array(),
            'gateway_branching_probabilities':
                [
                    gateway_probabilities.to_dict()
                    for gateway_probabilities in self.gateway_branching_probabilities
                ]
                if isinstance(self.gateway_branching_probabilities[0], GatewayProbabilities)
                else self.gateway_branching_probabilities,
        }

        if self.event_distribution:
            parameters['event_distribution'] = self.event_distribution

        return parameters

    def to_json_file(self, file_path: Path) -> None:
        """JSON compatible with Prosimos."""
        with file_path.open('w') as f:
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
    print_notice(f'Number of simulation cases: {settings.num_simulation_cases}')

    args = [
        'diff_res_bpsim', 'start-simulation',
        '--bpmn_path', settings.bpmn_path.__str__(),
        '--json_path', settings.parameters_path.__str__(),
        '--log_out_path', settings.output_log_path.__str__(),
        '--total_cases', str(settings.num_simulation_cases),
        '--starting_at', settings.simulation_start.isoformat()
    ]

    execute_shell_cmd(args)


def simulate_and_evaluate(
        model_path: Path,
        parameters_path: Path,
        output_dir: Path,
        simulation_cases: int,
        simulation_start_time: pd.Timestamp,
        validation_log: pd.DataFrame,
        validation_log_ids: EventLogIDs,
        metrics: List[Metric],
        num_simulations: int = 1
) -> List[dict]:
    """
    Simulates a process model using Prosimos num_simulations times in parallel.

    :param model_path: Path to the BPMN model.
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

    simulation_log_paths = simulate_in_parallel(model_path, num_simulations, output_dir, parameters_path,
                                                simulation_cases, simulation_start_time)

    evaluation_measurements = evaluate_logs(metrics, simulation_log_paths, validation_log, validation_log_ids)

    return evaluation_measurements


def simulate_in_parallel(
        model_path: Path,
        num_simulations: int,
        output_dir: Path,
        parameters_path: Path,
        simulation_cases: int,
        simulation_start_time: pd.Timestamp,
) -> List[Path]:
    """
    Simulates a process model using Prosimos num_simulations times in parallel.

    :param model_path: Path to the BPMN model.
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
            bpmn_path=model_path,
            parameters_path=parameters_path,
            output_log_path=output_dir / f'simulated_log_{rep}.csv',
            num_simulation_cases=simulation_cases,
            simulation_start=simulation_start_time,
        )
        for rep in range(num_simulations)]

    print_step(f'Simulating {len(simulation_arguments)} times with {w_count} workers')

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
        (simulation_log_paths[index], PROSIMOS_COLUMNS, index)
        for index in range(len(simulation_log_paths))
    ]

    print_step(f'Reading {len(read_arguments)} simulated logs with {w_count} workers')

    with Pool(w_count) as pool:
        simulated_logs = pool.map(_read_simulated_log, read_arguments)

    # Evaluate

    evaluation_arguments = [
        (validation_log, validation_log_ids, log, PROSIMOS_COLUMNS, metrics)
        for log in simulated_logs
    ]

    print_step(f'Evaluating {len(evaluation_arguments)} simulated logs with {w_count} workers')

    with Pool(w_count) as pool:
        evaluation_measurements = pool.map(_evaluate_logs_using_metrics, evaluation_arguments)
    evaluation_measurements = list(itertools.chain.from_iterable(evaluation_measurements))

    return evaluation_measurements


def _read_simulated_log(arguments: Tuple):
    log_path, log_ids, simulation_repetition_index = arguments

    df, _ = read(log_path, log_ids=log_ids)

    df['role'] = df['resource']
    df['source'] = 'simulation'
    df['run_num'] = simulation_repetition_index

    return df


def _evaluate_logs_using_metrics(arguments: Tuple) -> List[dict]:
    validation_log: pd.DataFrame = arguments[0]
    validation_log_ids: EventLogIDs = arguments[1]
    simulated_log: pd.DataFrame = arguments[2]
    simulated_log_ids: EventLogIDs = arguments[3]
    metrics: List[Metric] = arguments[4]

    rep = simulated_log.iloc[0].run_num

    measurements = []
    for metric in metrics:
        value = compute_metric(metric, validation_log, validation_log_ids, simulated_log, simulated_log_ids)
        measurements.append({'run_num': rep, 'metric': metric, 'value': value})

    return measurements

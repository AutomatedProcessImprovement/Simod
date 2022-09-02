import itertools
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from networkx import DiGraph
from tqdm import tqdm

from .simulation_parameters import gateway_probabilities
from simod.process_structure.simulation_parameters.activity_resources import ActivityResourceDistribution, ResourceDistribution
from simod.process_structure.simulation_parameters.calendars import Calendar
from simod.process_structure.simulation_parameters.distributions import Distribution
from simod.process_structure.simulation_parameters.gateway_probabilities import GatewayProbabilities
from simod.process_structure.simulation_parameters.resource_profiles import ResourceProfile
from ..analyzers.sim_evaluator import evaluate_logs
from ..cli_formatter import print_notice
from ..configuration import PDFMethod, GateManagement, Configuration, SimulatorKind
from ..discovery import inter_arrival_distribution
from ..discovery.tasks_evaluator import TaskEvaluator
from simod.event_log_processing.reader import EventLogReader
from ..event_log_processing.event_log_ids import EventLogIDs
from ..process_model.bpmn import BPMNReaderWriter
from ..support_utils import execute_shell_cmd, progress_bar_async

PROSIMOS_COLUMN_MAPPING = {  # TODO: replace with EventLogIDs
    'case_id': 'caseid',
    'activity': 'task',
    'enable_time': 'enabled_timestamp',
    'start_time': 'start_timestamp',
    'end_time': 'end_timestamp',
    'resource': 'user'
}


@dataclass
class Parameters:
    """Simulation parameters required by Prosimos."""
    resource_profiles: List[ResourceProfile]
    resource_calendars: List[Calendar]
    task_resource_distributions: List[ActivityResourceDistribution]
    arrival_distribution: Distribution
    arrival_calendar: Calendar
    gateway_branching_probabilities: List[GatewayProbabilities]

    def to_dict(self) -> dict:
        """Dictionary compatible with Prosimos."""
        return {
            'resource_profiles':
                [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            'resource_calendars':
                [calendar.to_dict() for calendar in self.resource_calendars],
            'task_resource_distribution':
                [activity_resources.to_dict() for activity_resources in self.task_resource_distributions],
            'arrival_time_distribution':
                self.arrival_distribution.to_dict(),
            'arrival_time_calendar':
                self.arrival_calendar.to_array(),
            'gateway_branching_probabilities':
                [gateway_probabilities.to_dict() for gateway_probabilities in self.gateway_branching_probabilities]
        }

    def to_json_file(self, file_path: Path) -> None:
        """JSON compatible with Prosimos."""
        with file_path.open('w') as f:
            json.dump(self.to_dict(), f)


def undifferentiated_resources_parameters(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph,
        pdf_method: PDFMethod,
        bpmn_reader: BPMNReaderWriter,
        gateways_probability_type: GateManagement) -> Parameters:
    calendar_24_7 = Calendar.all_day_long()

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_24_7.id)
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = [calendar_24_7]

    arrival_rate = inter_arrival_distribution.discover(process_graph, log, pdf_method)
    arrival_distribution = Distribution.from_simod_dict(arrival_rate)

    arrival_calendar = calendar_24_7

    gateway_probabilities_ = gateway_probabilities.discover(log, bpmn_path, gateways_probability_type)

    task_resource_distributions = _task_resource_distribution(
        log, process_graph, pdf_method, bpmn_reader, undifferentiated_resource_profile)

    return Parameters(
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        task_resource_distributions=task_resource_distributions,
        arrival_distribution=arrival_distribution,
        arrival_calendar=arrival_calendar,
        gateway_branching_probabilities=gateway_probabilities_
    )


def _task_resource_distribution(
        log: pd.DataFrame,
        process_graph: DiGraph,
        pdf_method: PDFMethod,
        bpmn_reader: BPMNReaderWriter,
        undifferentiated_resource_profile: ResourceProfile) -> List[ActivityResourceDistribution]:
    # extracting activities distribution
    log['role'] = 'SYSTEM'  # TaskEvaluator requires a role column
    resource_pool_metadata = {'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM'}
    activities_distributions = TaskEvaluator(process_graph, log, resource_pool_metadata, pdf_method).elements_data

    # activities' IDs and names from BPMN model
    activities_info = bpmn_reader.read_activities()

    task_resource_distributions = []
    normal_activities_bpmn_elements_ids = []

    # handling Start and End activities if present which always have fixed duration of 0
    for activity in activities_info:
        if activity['task_name'].lower() in ['start', 'end']:
            task_resource_distributions.append(
                ActivityResourceDistribution(
                    activity_id=activity['task_id'],
                    activity_resources_distributions=[
                        ResourceDistribution(resource_id=activity['task_name'], distribution=Distribution.fixed(0))
                    ]
                )
            )
        else:
            normal_activities_bpmn_elements_ids.append(activity['task_id'])

    normal_resources = list(
        filter(lambda r: r.name.lower() not in ['start', 'end'],
               undifferentiated_resource_profile.resources)
    )

    # handling other (normal) activities without Start and End
    for activity_id in normal_activities_bpmn_elements_ids:
        # getting activity distribution from BPMN
        activity_distribution: Optional[Distribution] = None
        for item in activities_distributions:
            if item['elementid'] == activity_id:
                distribution_data = {
                    'dname': item['type'],
                    'dparams': {
                        'mean': item['mean'],
                        'arg1': item['arg1'],
                        'arg2': item['arg2'],
                    }
                }
                activity_distribution = Distribution.from_simod_dict(distribution_data)
                break
        if activity_distribution is None:
            raise Exception(f'Distribution for activity {activity_id} not found')

        # in undifferentiated resources, all activities are assigned to each resource except Start and End,
        # Start and End have their own distinct resource
        resources_distributions = [
            ResourceDistribution(resource.id, activity_distribution)
            for resource in normal_resources
        ]

        task_resource_distributions.append(ActivityResourceDistribution(activity_id, resources_distributions))

    return task_resource_distributions


@dataclass
class ProsimosSettings:
    """Prosimos settings."""

    bpmn_path: Path
    parameters_path: Path
    output_log_path: Path
    num_simulation_cases: int

    @staticmethod
    def from_configuration(
            settings: Configuration,
            simulation_repetition_index: str,
            num_simulation_cases: Optional[int] = None) -> 'ProsimosSettings':
        bpmn_path = settings.output / (settings.project_name + '.bpmn')
        output_log_path = settings.output / 'sim_data' / f'{settings.project_name}_{simulation_repetition_index}.csv'
        parameters_path = bpmn_path.with_suffix('.json')
        if num_simulation_cases is None:
            num_simulation_cases = settings.simulation_cases
        else:
            num_simulation_cases = num_simulation_cases

        return ProsimosSettings(
            bpmn_path=bpmn_path,
            output_log_path=output_log_path,
            parameters_path=parameters_path,
            num_simulation_cases=num_simulation_cases,
        )


def _simulate_with_prosimos(settings: ProsimosSettings):
    print_notice(f'Prosimos simulator has been chosen')
    print_notice(f'Number of simulation cases: {settings.num_simulation_cases}')

    args = [
        'diff_res_bpsim', 'start-simulation',
        '--bpmn_path', settings.bpmn_path.__str__(),
        '--json_path', settings.parameters_path.__str__(),
        '--log_out_path', settings.output_log_path.__str__(),
        '--total_cases', str(settings.num_simulation_cases)
    ]

    execute_shell_cmd(args)


def simulate_undifferentiated(settings: Configuration, previous_step_result: Tuple, validation_log: pd.DataFrame):
    bpmn_path, json_path, simulation_cases = previous_step_result

    if settings.simulator is not SimulatorKind.CUSTOM:
        raise ValueError(f'Unknown simulator {settings.simulator}')

    num_simulations = settings.simulation_repetitions
    cpu_count = multiprocessing.cpu_count()
    w_count = num_simulations if num_simulations <= cpu_count else cpu_count
    pool = multiprocessing.Pool(processes=w_count)

    # Simulate
    simulation_arguments = [
        ProsimosSettings(
            bpmn_path=bpmn_path,
            parameters_path=json_path,
            output_log_path=settings.output / 'sim_data' / f'{settings.project_name}_{rep}.csv',
            num_simulation_cases=simulation_cases)
        for rep in range(num_simulations)]
    p = pool.map_async(_simulate_with_prosimos, simulation_arguments)
    progress_bar_async(p, 'simulating', num_simulations)

    # Read simulated logs
    read_arguments = [(simulation_arguments[index].output_log_path, PROSIMOS_COLUMN_MAPPING, index)
                      for index in range(num_simulations)]
    p = pool.map_async(_read_simulated_log, read_arguments)
    progress_bar_async(p, 'reading simulated logs', num_simulations)

    # Evaluate
    evaluation_arguments = [(settings, validation_log, log) for log in p.get()]
    if simulation_cases > 1000:
        pool.close()
        results = [evaluate_logs(arg) for arg in tqdm(evaluation_arguments, 'evaluating results')]
        evaluation_measurements = list(itertools.chain(*results))
    else:
        p = pool.map_async(evaluate_logs, evaluation_arguments)
        progress_bar_async(p, 'evaluating results', num_simulations)
        pool.close()
        evaluation_measurements = list(itertools.chain(*p.get()))

    return evaluation_measurements


def _read_simulated_log(arguments: Tuple):
    log_path, log_column_mapping, simulation_repetition_index = arguments

    reader = EventLogReader(log_path=log_path, column_names=log_column_mapping)

    reader.df.rename(columns={'user': 'resource'}, inplace=True)
    reader.df['role'] = reader.df['resource']
    reader.df['source'] = 'simulation'
    reader.df['run_num'] = simulation_repetition_index
    reader.df = reader.df[~reader.df.task.isin(['Start', 'End'])]

    return reader.df

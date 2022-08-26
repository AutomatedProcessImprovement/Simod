import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from networkx import DiGraph

from . import gateway_probabilities
from .activity_resources import ActivityResourceDistribution, ResourceDistribution
from .calendars import Calendar
from .distributions import Distribution
from .gateway_probabilities import GatewayProbabilities
from .resource_profiles import ResourceProfile
from ..configuration import PDFMethod, GateManagement
from ..discovery import inter_arrival_distribution
from ..discovery.tasks_evaluator import TaskEvaluator
from ..event_log import EventLogIDs
from ..readers.bpmn_reader import BpmnReader


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
                self.arrival_calendar.to_dict(),
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
        bpmn_reader: BpmnReader,
        gateways_probability_type: GateManagement) -> Parameters:
    calendar_24_7 = Calendar.all_day_long()

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_24_7.id)
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = [calendar_24_7]

    arrival_rate = inter_arrival_distribution.discover(process_graph, log, pdf_method)
    arrival_distribution = Distribution.from_simod_dict(arrival_rate)

    arrival_calendar = calendar_24_7

    gateway_probabilities_ = gateway_probabilities.discover(log, bpmn_path, gateways_probability_type)

    task_resource_distributions = __task_resource_distribution(
        log, process_graph, pdf_method, bpmn_reader, undifferentiated_resource_profile)

    return Parameters(
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        task_resource_distributions=task_resource_distributions,
        arrival_distribution=arrival_distribution,
        arrival_calendar=arrival_calendar,
        gateway_branching_probabilities=gateway_probabilities_
    )


def __task_resource_distribution(
        log: pd.DataFrame,
        process_graph: DiGraph,
        pdf_method: PDFMethod,
        bpmn_reader: BpmnReader,
        undifferentiated_resource_profile: ResourceProfile):
    log['role'] = 'SYSTEM'  # TaskEvaluator requires a role column
    resource_pool_metadata = {
        'id': 'QBP_DEFAULT_RESOURCE',
        'name': 'SYSTEM',
    }
    activities_distributions = TaskEvaluator(process_graph, log, resource_pool_metadata, pdf_method).elements_data

    activities_info = bpmn_reader.get_tasks_info()

    activities_bpmn_elements_ids = []
    for activity in activities_info:
        if activity['task_name'].lower() not in ['start', 'end']:
            activities_bpmn_elements_ids.append(activity['task_id'])

    task_resource_distributions = []
    for activity_id in activities_bpmn_elements_ids:
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

        # assigning activity distribution to all resources
        resources_distributions = []
        for resource in undifferentiated_resource_profile.resources:
            resources_distributions.append(
                ResourceDistribution(resource_id=resource.id, distribution=activity_distribution))

        task_resource_distributions.append(
            ActivityResourceDistribution(activity_id=activity_id,
                                         activity_resources_distributions=resources_distributions))

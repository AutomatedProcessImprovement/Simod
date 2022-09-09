from pathlib import Path
from typing import List, Optional

import pandas as pd
from networkx import DiGraph

from simod.simulation.parameters.activity_resources import ActivityResourceDistribution, ResourceDistribution
from simod.bpm.reader_writer import BPMNReaderWriter
from simod.configuration import PDFMethod, GateManagement
from simod.discovery import inter_arrival_distribution
from simod.discovery.tasks_evaluator import TaskEvaluator
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.parameters.calendars import Calendar
from simod.simulation.parameters.distributions import Distribution
from simod.simulation.parameters.gateway_probabilities import mine_gateway_probabilities
from simod.simulation.parameters.resource_profiles import ResourceProfile
from simod.simulation.prosimos import SimulationParameters


def mine_simulation_parameters_default_24_7(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph,
        pdf_method: PDFMethod,
        bpmn_reader: BPMNReaderWriter,
        gateways_probability_type: GateManagement) -> SimulationParameters:
    calendar_24_7 = Calendar.all_day_long()

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_24_7.id)
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = [calendar_24_7]

    arrival_rate = inter_arrival_distribution.discover(process_graph, log, pdf_method)
    arrival_distribution = Distribution.from_simod_dict(arrival_rate)

    arrival_calendar = calendar_24_7

    gateway_probabilities_ = mine_gateway_probabilities(log, bpmn_path, gateways_probability_type)

    task_resource_distributions = _task_resource_distribution(
        log, process_graph, pdf_method, bpmn_reader, undifferentiated_resource_profile)

    return SimulationParameters(
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



from pathlib import Path
from typing import List, Tuple

import pandas as pd
from networkx import DiGraph

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_notice
from simod.configuration import GatewayProbabilitiesDiscoveryMethod, CalendarType, PDFMethod, CalendarSettings
from simod.discovery import inter_arrival_distribution
from simod.discovery.distribution import get_best_distribution
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.calendar_discovery import case_arrival, resource as resource_calendar
from simod.simulation.parameters.activity_resources import ActivityResourceDistribution, ResourceDistribution
from simod.simulation.parameters.calendars import Calendar
from simod.simulation.parameters.distributions import Distribution
from simod.simulation.parameters.gateway_probabilities import mine_gateway_probabilities
from simod.simulation.parameters.intervals import Interval, prosimos_interval_to_interval, pd_intervals_to_intervals, \
    intersect_intervals
from simod.simulation.parameters.resource_profiles import ResourceProfile
from simod.simulation.prosimos import SimulationParameters


def mine_parameters(
        case_arrival_settings: CalendarSettings,
        resource_profiles_settings: CalendarSettings,
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        model_path: Path,
        gateways_probability_type: GatewayProbabilitiesDiscoveryMethod
) -> SimulationParameters:
    """Mine simulation parameters given the settings for resources and case arrival."""

    gateway_probabilities = mine_gateway_probabilities(log, log_ids, model_path, gateways_probability_type)

    # Case arrival parameters

    case_arrival_discovery_type = case_arrival_settings.discovery_type
    if case_arrival_discovery_type == CalendarType.DEFAULT_24_7:
        arrival_calendar = Calendar.all_day_long()
    elif case_arrival_discovery_type == CalendarType.DEFAULT_9_5:
        arrival_calendar = Calendar.work_day()
    # NOTE: discarding other types of discovery for case arrival
    elif case_arrival_discovery_type in (CalendarType.UNDIFFERENTIATED,
                                         CalendarType.DIFFERENTIATED_BY_POOL,
                                         CalendarType.DIFFERENTIATED_BY_RESOURCE):
        arrival_calendar = case_arrival.discover_undifferentiated(
            log,
            log_ids,
            granularity=case_arrival_settings.granularity,
            min_confidence=case_arrival_settings.confidence,
            desired_support=case_arrival_settings.support,
            min_participation=case_arrival_settings.participation,
        )
    else:
        raise ValueError(f'Unknown calendar discovery type: {case_arrival_discovery_type}')

    bpmn_reader = BPMNReaderWriter(model_path)
    process_graph = bpmn_reader.as_graph()

    pdf_method = PDFMethod.AUTOMATIC
    arrival_distribution = inter_arrival_distribution.discover(process_graph, log, log_ids, pdf_method)

    # Resource parameters

    resource_discovery_type = resource_profiles_settings.discovery_type
    if resource_discovery_type == CalendarType.DEFAULT_24_7:
        resource_profiles, resource_calendars, task_resource_distributions = mine_default_for_resources(
            log, log_ids, model_path, process_graph, Calendar.all_day_long())
    elif resource_discovery_type == CalendarType.DEFAULT_9_5:
        resource_profiles, resource_calendars, task_resource_distributions = mine_default_for_resources(
            log, log_ids, model_path, process_graph, Calendar.work_day())
    elif resource_discovery_type == CalendarType.UNDIFFERENTIATED:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_undifferentiated(
            log, log_ids, model_path, process_graph)
    elif resource_discovery_type == CalendarType.DIFFERENTIATED_BY_POOL:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_pools(
            log, log_ids, process_graph)
    elif resource_discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_differentiated(
            log, log_ids, model_path, process_graph)
    else:
        raise ValueError(f'Unknown calendar discovery type: {resource_discovery_type}')

    return SimulationParameters(
        gateway_branching_probabilities=gateway_probabilities,
        arrival_calendar=arrival_calendar,
        arrival_distribution=arrival_distribution,
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        task_resource_distributions=task_resource_distributions,
    )


def mine_default_24_7(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph,
        pdf_method: PDFMethod,
        bpmn_reader: BPMNReaderWriter,
        gateways_probability_type: GatewayProbabilitiesDiscoveryMethod) -> SimulationParameters:
    """Simulation parameters with default calendar 24/7."""

    calendar_24_7 = Calendar.all_day_long()

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar_24_7.id)
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = [calendar_24_7]

    arrival_distribution = inter_arrival_distribution.discover(process_graph, log, log_ids, pdf_method)

    arrival_calendar = calendar_24_7

    gateway_probabilities_ = mine_gateway_probabilities(log, log_ids, bpmn_path, gateways_probability_type)

    activity_duration_distributions = _activity_duration_distributions_undifferentiated(
        log, log_ids, process_graph)

    return SimulationParameters(
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        task_resource_distributions=activity_duration_distributions,
        arrival_distribution=arrival_distribution,
        arrival_calendar=arrival_calendar,
        gateway_branching_probabilities=gateway_probabilities_
    )


def mine_default_for_resources(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph,
        calendar: Calendar) -> Tuple:
    """Simulation parameters with default calendar 24/7."""

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(log, log_ids, bpmn_path, calendar.id)
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = [calendar]

    activity_duration_distributions = _activity_duration_distributions_undifferentiated(log, log_ids, process_graph)

    return resource_profiles, resource_calendars, activity_duration_distributions


def _resource_parameters_for_undifferentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph) -> Tuple:
    """Simulation parameters with undifferentiated resources."""

    calendars = [resource_calendar.discover_undifferentiated(log, log_ids)]
    assert len(calendars) == 1, "Only one resource calendar is supported for undifferentiated resources."

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(
        log, log_ids, bpmn_path, calendars[0].id)
    resource_profiles = [undifferentiated_resource_profile]

    task_resource_distributions = _activity_duration_distributions_undifferentiated(log, log_ids, process_graph)

    return resource_profiles, calendars, task_resource_distributions


def _resource_parameters_for_pools(log: pd.DataFrame, log_ids: EventLogIDs, process_graph: DiGraph) -> Tuple:
    """Simulation parameters for resource pools."""

    calendars, pool_mapping = resource_calendar.discover_per_resource_pool(log, log_ids)
    assert len(calendars) > 0, "At least one resource calendar is required for resource pools."

    pool_profiles = ResourceProfile.differentiated_by_pool_(log, log_ids, pool_mapping, process_graph)

    activity_duration_distributions = _activity_duration_distributions_pools(log, log_ids, process_graph, pool_mapping)

    return pool_profiles, calendars, activity_duration_distributions


def _resource_parameters_for_differentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph) -> Tuple:
    """Simulation parameters for fully differentiated resources."""

    resource_calendars = resource_calendar.discover_per_resource(log, log_ids)
    assert len(resource_calendars) > 0, "At least one resource calendar is required."

    resource_profiles = ResourceProfile.differentiated_by_resource(log, log_ids, bpmn_path, resource_calendars)

    activity_duration_distributions = _activity_duration_distributions_differentiated(
        log, log_ids, process_graph, resource_calendars)

    return resource_profiles, resource_calendars, activity_duration_distributions


def _activity_duration_distributions_differentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        process_graph: DiGraph,
        calendars: List[Calendar]) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for fully differentiated resources for the Prosimos simulator.
    """
    # Finding the best distributions for each activity-resource pair
    activity_duration_distributions = {}
    for (activity, resource_), group in log.groupby([log_ids.activity, log_ids.resource]):
        # Naive approach
        #
        # durations = (group[log_ids.end_time] - group[log_ids.start_time]).to_list()
        # durations = [duration.total_seconds() for duration in durations]

        calendar = next((calendar for calendar in calendars if calendar.id == resource_), None)
        assert calendar is not None, f"Resource calendar for resource {resource_} not found."

        durations = _get_activity_durations_without_off_duty(group, log_ids, calendar)
        if len(durations) == 0:
            durations = [0]
            print_notice(f"No durations for activity {activity} and resource {resource_}. "
                         f"Setting the activity duration distribution to fixed(0).")

        # Computing the distribution

        distribution = get_best_distribution(durations)

        if activity not in activity_duration_distributions:
            activity_duration_distributions[activity] = {resource_: distribution}
        else:
            activity_duration_distributions[activity][resource_] = distribution

    # Getting activities' IDs from the model graph
    activity_ids = get_activities_ids_by_name(process_graph)

    # Collecting the distributions
    distributions = []
    for activity_name in activity_duration_distributions:
        resources_distributions = [
            ResourceDistribution(resource_, activity_duration_distributions[activity_name][resource_])
            for resource_ in activity_duration_distributions[activity_name]
        ]

        activity_id = activity_ids[activity_name]
        distributions.append(ActivityResourceDistribution(activity_id, resources_distributions))

    return distributions


def _get_activity_durations_without_off_duty(df: pd.DataFrame, log_ids: EventLogIDs, calendar: Calendar) -> List[float]:
    """
    Returns activity durations without off-duty time.
    """
    activity_intervals = _get_activity_intervals(df, log_ids)
    overlapping_intervals = _get_overlapping_intervals(activity_intervals, calendar)
    overlapping_durations = [interval.duration().total_seconds() for interval in overlapping_intervals]
    return overlapping_durations


def _get_activity_intervals(df: pd.DataFrame, log_ids: EventLogIDs) -> List[pd.Interval]:
    """
    Returns a list of activity intervals.
    """
    start_times = df[log_ids.start_time].to_list()
    end_times = df[log_ids.end_time].to_list()
    return [pd.Interval(start, end) for start, end in zip(start_times, end_times)]


def _get_overlapping_intervals(intervals: List[pd.Interval], calendar: Calendar) -> List[Interval]:
    """
    Returns a list of intervals that overlap with the calendar.
    """
    calendar_intervals = [prosimos_interval_to_interval(timetable.to_dict()) for timetable in calendar.timetables]
    intervals_ = pd_intervals_to_intervals(intervals)
    overlapping_intervals = intersect_intervals(intervals_, calendar_intervals)
    return overlapping_intervals


def _activity_duration_distributions_pools(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        process_graph: DiGraph,
        pool_by_resource_name: dict) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for pooled resources for the Prosimos simulator.
    """

    # Adding pool information to the log
    pool_key = 'pool'
    pool_data = pd.DataFrame(pool_by_resource_name.items(), columns=[log_ids.resource, pool_key])
    log = log.merge(pool_data, on=log_ids.resource)

    # Finding the best distributions for each activity-pool pair
    activity_duration_distributions = {}
    for (activity, pool_name), group in log.groupby([log_ids.activity, pool_key]):
        durations = (group[log_ids.end_time] - group[log_ids.start_time]).to_list()
        durations = [duration.total_seconds() for duration in durations]
        distribution = get_best_distribution(durations)

        if activity not in activity_duration_distributions:
            activity_duration_distributions[activity] = {pool_name: distribution}
        else:
            activity_duration_distributions[activity][pool_name] = distribution

    # Getting activities' IDs from the model graph
    activity_ids = get_activities_ids_by_name(process_graph)

    # Collecting the distributions for Prosimos
    distributions = []
    for activity_name in activity_ids:
        activity_id = activity_ids[activity_name]
        activity_resources = log[log[log_ids.activity] == activity_name][log_ids.resource].unique()
        resources_ = []
        for resource in activity_resources:
            pool = pool_by_resource_name[resource]
            distribution = activity_duration_distributions[activity_name][pool]
            resources_.append(ResourceDistribution(resource, distribution))
        distributions.append(ActivityResourceDistribution(activity_id, resources_))

    return distributions


def _activity_duration_distributions_undifferentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        process_graph: DiGraph) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for undifferentiated resources for the Prosimos simulator.
    """

    # Finding the best distribution for all activities durations
    durations = (log[log_ids.end_time] - log[log_ids.start_time]).to_list()
    durations = [duration.total_seconds() for duration in durations]
    activity_duration_distribution = get_best_distribution(durations)

    # Getting activities' IDs from the model graph
    activity_ids = get_activities_ids_by_name(process_graph)

    # Resources without Start and End
    resources = log[log_ids.resource].unique()
    resources_filtered = list(filter(lambda r: r.lower() not in ['start', 'end'], resources))

    # Collecting the distributions
    task_resource_distributions = []
    for name in activity_ids:
        id_ = activity_ids[name]

        if name.lower() in ('start', 'end'):
            task_resource_distributions.append(
                ActivityResourceDistribution(
                    activity_id=id_,
                    activity_resources_distributions=[
                        ResourceDistribution(resource_id=name, distribution=Distribution.fixed(0))
                    ]
                )
            )
        else:
            resources_distributions = [
                ResourceDistribution(resource_, activity_duration_distribution)
                for resource_ in resources_filtered
            ]

            task_resource_distributions.append(ActivityResourceDistribution(id_, resources_distributions))

    return task_resource_distributions


def get_activities_ids_by_name(process_graph: DiGraph) -> dict:
    """Returns activities' IDs from the model graph"""
    model_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()), orient='index')
    model_data = model_data[model_data.type.isin(['task', 'start', 'end'])]
    items = model_data[['name', 'id']].to_records(index=False)
    return {item[0]: item[1] for item in items}  # {name: id}

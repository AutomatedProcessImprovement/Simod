from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
from networkx import DiGraph
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.case_arrival import discover_case_arrival_calendar, discover_inter_arrival_distribution
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod, GatewayProbabilities, \
    compute_gateway_probabilities
from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.log_ids import EventLogIDs
from pix_framework.statistics.distribution import get_best_fitting_distribution

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.cli_formatter import print_notice
from simod.settings.temporal_settings import CalendarSettings, CalendarType
from simod.simulation.calendar_discovery import resource as resource_calendar
from simod.simulation.calendar_discovery.resource import full_day_schedule, working_hours_schedule, \
    UNDIFFERENTIATED_RESOURCE_POOL_KEY
from simod.simulation.parameters.intervals import Interval, intersect_intervals, prosimos_interval_to_interval_safe, \
    pd_interval_to_interval
from simod.simulation.parameters.resource_activity_performances import ActivityResourceDistribution, \
    ResourceDistribution
from simod.simulation.parameters.resource_profiles import ResourceProfile
from simod.simulation.prosimos import SimulationParameters


def mine_parameters(
        case_arrival_settings: CalendarSettings,
        resource_profiles_settings: CalendarSettings,
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        model_path: Path,
        gateways_probability_method: Optional[GatewayProbabilitiesDiscoveryMethod] = None,
        gateway_probabilities: Optional[List[GatewayProbabilities]] = None,
        process_graph: Optional[DiGraph] = None,
) -> SimulationParameters:
    """
    Mine simulation parameters given the settings for resources and case arrival.
    """
    # Gateway probabilities
    bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
    if gateway_probabilities is None:
        assert gateways_probability_method is not None, \
            "Either gateway probabilities or a method to mine them must be provided."
        gateway_probabilities = compute_gateway_probabilities(log, log_ids, bpmn_graph, gateways_probability_method)

    if not process_graph:
        bpmn_reader = BPMNReaderWriter(model_path)
        process_graph = bpmn_reader.as_graph()

    # Case arrival parameters
    arrival_calendar = discover_case_arrival_calendar(
        log,
        log_ids,
        granularity=case_arrival_settings.granularity
    )
    arrival_distribution = discover_inter_arrival_distribution(log, log_ids)

    # Resource parameters

    resource_discovery_type = resource_profiles_settings.discovery_type
    if resource_discovery_type == CalendarType.DEFAULT_24_7:
        resource_profiles, resource_calendars, task_resource_distributions = mine_default_for_resources(
            log, log_ids, model_path, process_graph, full_day_schedule()
        )
    elif resource_discovery_type == CalendarType.DEFAULT_9_5:
        resource_profiles, resource_calendars, task_resource_distributions = mine_default_for_resources(
            log, log_ids, model_path, process_graph, working_hours_schedule()
        )
    elif resource_discovery_type == CalendarType.UNDIFFERENTIATED:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_undifferentiated(
            log, log_ids, model_path, process_graph
        )
    elif resource_discovery_type == CalendarType.DIFFERENTIATED_BY_POOL:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_pools(
            log, log_ids, process_graph
        )
    elif resource_discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE:
        resource_profiles, resource_calendars, task_resource_distributions = _resource_parameters_for_differentiated(
            log, log_ids, model_path, process_graph
        )
    else:
        raise ValueError(f"Unknown calendar discovery type: {resource_discovery_type}")

    assert len(resource_profiles) > 0, "No resource profiles found"
    assert len(resource_calendars) > 0, "No resource calendars found"
    assert len(task_resource_distributions) > 0, "No task resource distributions found"

    parameters = SimulationParameters(
        gateway_branching_probabilities=gateway_probabilities,
        arrival_calendar=arrival_calendar,
        arrival_distribution=arrival_distribution,
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        task_resource_distributions=task_resource_distributions,
        event_distribution=None,
    )

    return parameters


def mine_default_for_resources(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph,
        calendar: RCalendar,
) -> Tuple[List[ResourceProfile], Dict[str, RCalendar], List[ActivityResourceDistribution]]:
    """Simulation parameters with default calendar 24/7."""

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(
        log, log_ids, bpmn_path, calendar.calendar_id
    )
    resource_profiles = [undifferentiated_resource_profile]

    resource_calendars = {'undifferentiated': calendar}

    activity_duration_distributions = _activity_duration_distributions_undifferentiated(
        log, log_ids, process_graph, calendar
    )

    return resource_profiles, resource_calendars, activity_duration_distributions


def _resource_parameters_for_undifferentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph
) -> Tuple[List[ResourceProfile], Dict[str, RCalendar], List[ActivityResourceDistribution]]:
    """
    Simulation parameters with undifferentiated resources.
    """
    resource_name = UNDIFFERENTIATED_RESOURCE_POOL_KEY

    calendars = resource_calendar.discover_undifferentiated(log, log_ids)

    undifferentiated_resource_profile = ResourceProfile.undifferentiated(
        log, log_ids, bpmn_path, calendars[resource_name].calendar_id
    )
    resource_profiles = [undifferentiated_resource_profile]

    task_resource_distributions = _activity_duration_distributions_undifferentiated(
        log, log_ids, process_graph, calendars[resource_name]
    )

    return resource_profiles, calendars, task_resource_distributions


def _resource_parameters_for_pools(log: pd.DataFrame, log_ids: EventLogIDs, process_graph: DiGraph) -> Tuple:
    """Simulation parameters for resource pools."""

    calendars, pool_mapping = resource_calendar.discover_per_resource_pool(log, log_ids)
    assert len(calendars) > 0, "At least one resource calendar is required for resource pools."

    # Adding pool information to the log
    pool_data = pd.DataFrame(pool_mapping.items(), columns=[log_ids.resource, 'pool'])
    log = log.merge(pool_data, on=log_ids.resource)

    # Adding calendar names to log by pool
    log = log.merge(
        pd.DataFrame.from_dict(
            {'pool': list(calendars.keys()),
             'resource_calendar': list(map(lambda c: c.calendar_id, list(calendars.values())))}
        ),
        on='pool',
        how='left'
    )

    pool_profiles = ResourceProfile.differentiated_by_pool_(
        log, log_ids, pool_mapping, process_graph
    )

    activity_duration_distributions = _activity_duration_distributions_pools(
        log, log_ids, process_graph, pool_mapping, calendars
    )

    return pool_profiles, calendars, activity_duration_distributions


def _resource_parameters_for_differentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        process_graph: DiGraph
) -> Tuple[List[ResourceProfile], Dict[str, RCalendar], List[ActivityResourceDistribution]]:
    """Simulation parameters for fully differentiated resources."""

    resource_calendars = resource_calendar.discover_per_resource(log, log_ids)
    assert len(resource_calendars) > 0, "At least one resource calendar is required."

    resource_profiles = ResourceProfile.differentiated_by_resource(
        log, log_ids, bpmn_path, resource_calendars
    )

    activity_duration_distributions = _activity_duration_distributions_differentiated(
        log, log_ids, process_graph, resource_calendars
    )

    return resource_profiles, resource_calendars, activity_duration_distributions


def _activity_duration_distributions_differentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        process_graph: DiGraph,
        resource_calendars: Dict[str, RCalendar],  # resource_name -> calendar
) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for fully differentiated resources for the Prosimos simulator.
    """
    # Finding the best distributions for each activity-resource pair
    activity_duration_distributions = {}
    for (activity, resource_), group in log.groupby([log_ids.activity, log_ids.resource]):
        calendar = resource_calendars[resource_]
        durations = _get_activity_durations_without_off_duty(group, log_ids, calendar.to_json())
        if len(durations) == 0:
            durations = [0]
            print_notice(
                f"No durations for activity {activity} and resource {resource_}. "
                f"Setting the activity duration distribution to fixed(0)."
            )

        # Computing the distribution

        distribution = get_best_fitting_distribution(
            durations
        ).to_prosimos_distribution()

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
            ResourceDistribution(
                resource_,
                activity_duration_distributions[activity_name][resource_]
            )
            for resource_ in activity_duration_distributions[activity_name]
        ]

        activity_id = activity_ids[activity_name]
        distributions.append(
            ActivityResourceDistribution(activity_id, resources_distributions)
        )

    return distributions


def _get_activity_durations_without_off_duty(
        df: pd.DataFrame, log_ids: EventLogIDs, timetables: List[dict]
) -> List[float]:
    """
    Returns activity durations without off-duty time.
    """
    activity_intervals = _get_activity_intervals(df, log_ids)
    overlapping_intervals = _get_overlapping_intervals(activity_intervals, timetables)
    overlapping_durations = []
    for intervals in overlapping_intervals:
        duration_intervals = [
            interval.duration().total_seconds() for interval in intervals
        ]
        overlapping_durations.append(sum(duration_intervals))

    return overlapping_durations


def _get_activity_intervals(df: pd.DataFrame, log_ids: EventLogIDs) -> List[pd.Interval]:
    """
    Returns a list of activity intervals.
    """
    start_times = df[log_ids.start_time].to_list()
    end_times = df[log_ids.end_time].to_list()
    return [pd.Interval(start, end) for start, end in zip(start_times, end_times)]


def _get_overlapping_intervals(intervals: List[pd.Interval], timetables: List[dict]) -> List[List[Interval]]:
    """
    Returns a list of lists of intervals that overlap with the calendar. First level of the list has intervals for each
    activity. So, each activity can have 1+ intervals.
    """
    calendar_intervals = []
    for timetable in timetables:
        calendar_intervals.extend(
            prosimos_interval_to_interval_safe(timetable)
        )

    intervals_ = [
        pd_interval_to_interval(interval)
        for interval in sorted(intervals, key=lambda item: item.left)
    ]

    overlapping_intervals = []
    for interval in intervals_:
        overlapping_intervals.append(intersect_intervals(interval, calendar_intervals))

    return overlapping_intervals


def _activity_duration_distributions_pools(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        process_graph: DiGraph,
        pool_by_resource_name: dict,
        resource_calendars: Dict[str, RCalendar],  # resource_name -> calendar
) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for pooled resources for the Prosimos simulator.
    """

    pool_key = "pool"

    # Finding the best distributions for each activity-pool pair
    activity_duration_distributions = {}
    for (activity, pool_name), group in log.groupby([log_ids.activity, pool_key]):
        calendar = resource_calendars[pool_name]
        durations = _get_activity_durations_without_off_duty(group, log_ids, calendar.to_json())
        if len(durations) == 0:
            durations = [0]
            print_notice(
                f"No durations for activity {activity} and pool {pool_name}. "
                f"Setting the activity duration distribution to fixed(0)."
            )

        distribution = get_best_fitting_distribution(durations).to_prosimos_distribution()

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
        process_graph: DiGraph,
        resource_calendar: RCalendar,
) -> List[ActivityResourceDistribution]:
    """
    Mines activity duration distributions for undifferentiated resources for the Prosimos simulator.
    """

    # Finding the best distribution for all activities durations

    activity_duration_distributions = {}
    for activity, group in log.groupby(log_ids.activity):
        durations = _get_activity_durations_without_off_duty(group, log_ids, resource_calendar.to_json())
        if len(durations) == 0:
            durations = [0]
            print_notice(
                f"No durations for activity {activity}. "
                f"Setting the activity duration distribution to fixed(0)."
            )

        activity_duration_distribution = get_best_fitting_distribution(
            durations
        ).to_prosimos_distribution()

        activity_duration_distributions[activity] = activity_duration_distribution

    # Getting activities' IDs from the model graph
    activity_ids = get_activities_ids_by_name(process_graph)

    # Resources without Start and End
    resources = log[log_ids.resource].unique()
    resources_filtered = list(
        filter(lambda r: r.lower() not in ["start", "end"], resources)
    )

    # Collecting the distributions
    distributions = []
    for name in activity_ids:
        id_ = activity_ids[name]

        if name.lower() in ("start", "end"):
            distributions.append(
                ActivityResourceDistribution(
                    activity_id=id_,
                    activity_resources_distributions=[
                        ResourceDistribution(
                            resource_id=name,
                            distribution={
                                "distribution_name": "fix",
                                "distribution_params": [{"value": 0}],
                            },
                        )
                    ],
                )
            )
        else:
            activity_duration_distribution = activity_duration_distributions[name]
            resources_distributions = [
                ResourceDistribution(resource_, activity_duration_distribution)
                for resource_ in resources_filtered
            ]

            distributions.append(
                ActivityResourceDistribution(id_, resources_distributions)
            )

    return distributions


def get_activities_ids_by_name(process_graph: DiGraph) -> dict:
    """Returns activities' IDs from the model graph"""
    model_data = pd.DataFrame.from_dict(
        dict(process_graph.nodes.data()), orient="index"
    )
    model_data = model_data[model_data.type.isin(["task", "start", "end"])]
    items = model_data[["name", "id"]].to_records(index=False)
    return {item[0]: item[1] for item in items}  # {name: id}

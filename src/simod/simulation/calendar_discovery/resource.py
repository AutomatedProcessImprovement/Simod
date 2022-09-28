from typing import NewType, Dict, List, Optional, Tuple, Union

import pandas as pd
from bpdfr_simulation_engine.resource_calendar import CalendarFactory

from simod.configuration import CalendarType
from simod.discovery.resource_pool_discoverer import ResourcePoolDiscoverer
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.parameters.calendars import Calendar

UNDIFFERENTIATED_RESOURCE_POOL_KEY = "undifferentiated_resource_pool"
RESOURCE_KEY = "org:resource"
ACTIVITY_KEY = "concept:name"
START_TIMESTAMP_KEY = "start_timestamp"
END_TIMESTAMP_KEY = "time:timestamp"

# TODO: Include min_participation into configuration and hyperopt
# TODO: Use existing hyperopt for confidence and support
# TODO: Update the configuration: res_cal_met, arr_cal_met, ...
# TODO: move calendar discovery code from Prosimos into Simod

PoolName = NewType('PoolName', str)
ResourceName = NewType('ResourceName', str)
PoolMapping = NewType('PoolMapping', Dict[ResourceName, PoolName])


def __resource_pool_analyser_result_to_pool_mapping(resource_table: List[dict]) -> PoolMapping:
    """
    Converts the result of ResourcePoolAnalyser to a dictionary mapping of resource names to pool names.
    :param resource_table: The result of ResourcePoolAnalyser.
    :return: PoolMapping.
    """
    mapping = {}
    for item in resource_table:
        pool = PoolName(item['role'])
        resource = ResourceName(item['resource'])
        mapping[resource] = pool
    return PoolMapping(mapping)


def _get_simod_column_names(keys: List[str], mapping: dict = None) -> Tuple:
    if not mapping:
        return tuple(keys)
    result = [mapping[key] for key in keys]
    return tuple(result)


def _discover_timetables(event_log: pd.DataFrame,
                         log_ids: EventLogIDs,
                         granularity=60,
                         min_confidence=0.1,
                         desired_support=0.7,
                         min_participation=0.4,
                         differentiated=True,
                         pool_mapping: Optional[PoolMapping] = None) -> dict:
    """
    Creates a calendar for the given event log using Prosimos. If the amount of events is too low, the results are not
    trustworthy. It's recommended to build a resource calendar for the whole resource pool instead of a single resource.
    The more the events there is in the log, the smaller the granularity should be.
    :param event_log: The event log to use.
    :param granularity: The number of minutes that is added to the start timestamp.
    :param min_confidence: The minimum confidence.
    :param desired_support: The desired support.
    :param min_participation: The minimum participation.
    :param differentiated: Whether to mine differentiated calendars for each resource or to use a single resource pool for all resources.
    :param pool_mapping: A dictionary mapping of resource names to pool names.
    :param columns_mapping: A dictionary mapping of column names.
    :return: the calendar dictionary with the resource names as keys and the working time intervals as values.
    """
    calendar_factory = CalendarFactory(granularity)
    for (index, event) in event_log.iterrows():
        if differentiated:
            if pool_mapping:
                resource = pool_mapping[event[log_ids.resource]]
            else:
                resource = event[log_ids.resource]
        else:
            resource = UNDIFFERENTIATED_RESOURCE_POOL_KEY

        activity = event[log_ids.activity]
        start_time = event[START_TIMESTAMP_KEY]
        end_time = event[log_ids.end_time]
        calendar_factory.check_date_time(resource, activity, start_time)
        calendar_factory.check_date_time(resource, activity, end_time)
    calendar_candidates = calendar_factory.build_weekly_calendars(min_confidence, desired_support, min_participation)
    calendar = {}
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            calendar[resource_id] = calendar_candidates[resource_id].to_json()
    return calendar


def discover_undifferentiated(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4) -> Calendar:
    timetables = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, False)
    timetables = timetables[UNDIFFERENTIATED_RESOURCE_POOL_KEY]

    calendar = Calendar(
        id='Undifferentiated_resources',
        name='Undifferentiated_resources',
        timetables=timetables
    )

    return calendar


def discover_per_resource_pool(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4) -> Tuple[List[Calendar], PoolMapping]:
    analyzer = ResourcePoolDiscoverer(
        event_log[[log_ids.activity, log_ids.resource]],
        activity_key=log_ids.activity,
        resource_key=log_ids.resource)

    pool_mapping = __resource_pool_analyser_result_to_pool_mapping(analyzer.resource_table)

    timetables_per_pool = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True, pool_mapping)

    calendars = [
        Calendar(id=pool_name, name=pool_name, timetables=timetables_per_pool[pool_name])
        for pool_name in timetables_per_pool
    ]

    return calendars, pool_mapping


def discover_per_resource(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4) -> List[Calendar]:
    timetables_per_pool = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True)

    calendars = [
        Calendar(id=pool_name, name=pool_name, timetables=timetables_per_pool[pool_name])
        for pool_name in timetables_per_pool
    ]

    return calendars


def discover(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        calendar_type: CalendarType,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4) -> Union[Calendar, List[Calendar]]:
    if calendar_type == CalendarType.UNDIFFERENTIATED:
        return discover_undifferentiated(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation)
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_POOL:
        return discover_per_resource_pool(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation)
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_RESOURCE:
        return discover_per_resource(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation)
    else:
        raise ValueError("Unknown calendar type.")

from typing import NewType, Dict, List, Optional, Tuple

import pandas as pd

from bpdfr_simulation_engine.resource_calendar import CalendarFactory
from simod.configuration import CalendarType
from simod.extraction.role_discovery import ResourcePoolAnalyser

UNDIFFERENTIATED_RESOURCE_POOL_KEY = "undifferentiated_resource_pool"
RESOURCE_KEY = "org:resource"
ACTIVITY_KEY = "concept:name"
START_TIMESTAMP_KEY = "start_timestamp"
END_TIMESTAMP_KEY = "time:timestamp"

# TODO: Include min_participation into configuration and hyperopt
# TODO: Use existing hyperopt for confidence and support
# TODO: Update the configuration: res_cal_met, arr_cal_met, ...

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


def make(event_log: pd.DataFrame,
         granularity=60,
         min_confidence=0.1,
         desired_support=0.7,
         min_participation=0.4,
         differentiated=True,
         pool_mapping: Optional[PoolMapping] = None,
         columns_mapping: dict = None) -> dict:
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
        resource_key, activity_key, end_time_key = _get_simod_column_names(
            keys=[RESOURCE_KEY, ACTIVITY_KEY, END_TIMESTAMP_KEY],
            mapping=columns_mapping)

        if differentiated:
            if pool_mapping:
                resource = pool_mapping[event[resource_key]]
            else:
                resource = event[resource_key]
        else:
            resource = UNDIFFERENTIATED_RESOURCE_POOL_KEY

        activity = event[activity_key]
        start_time = event[START_TIMESTAMP_KEY]
        end_time = event[end_time_key]
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
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        columns_mapping: dict = None) -> dict:
    return make(event_log, granularity, min_confidence, desired_support, min_participation, False, columns_mapping=columns_mapping)


def discover_per_resource_pool(
        event_log: pd.DataFrame,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        columns_mapping: dict = None) -> dict:
    analyzer = ResourcePoolAnalyser(
        event_log[[ACTIVITY_KEY, RESOURCE_KEY]], activity_key=ACTIVITY_KEY, resource_key=RESOURCE_KEY)
    pool_mapping = __resource_pool_analyser_result_to_pool_mapping(analyzer.resource_table)
    return make(event_log, granularity, min_confidence, desired_support, min_participation, True, pool_mapping, columns_mapping=columns_mapping)


def discover_per_resource(
        event_log: pd.DataFrame,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        columns_mapping: dict = None) -> dict:
    return make(event_log, granularity, min_confidence, desired_support, min_participation, True, columns_mapping=columns_mapping)


def discover(
        event_log: pd.DataFrame,
        calendar_type: CalendarType,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        columns_mapping: dict = None) -> dict:
    if calendar_type == CalendarType.UNDIFFERENTIATED:
        return discover_undifferentiated(event_log, granularity, min_confidence, desired_support, min_participation,
                                         columns_mapping)
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_POOL:
        return discover_per_resource_pool(event_log, granularity, min_confidence, desired_support, min_participation,
                                          columns_mapping)
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_RESOURCE:
        return discover_per_resource(event_log, granularity, min_confidence, desired_support, min_participation,
                                     columns_mapping)
    else:
        raise ValueError("Unknown calendar type.")

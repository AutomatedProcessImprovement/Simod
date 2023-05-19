from typing import NewType, Dict, List, Optional, Tuple

import pandas as pd
from pix_framework.log_ids import EventLogIDs
from prosimos.resource_calendar import CalendarFactory, RCalendar

from simod.discovery.resource_pool_discoverer import ResourcePoolDiscoverer

UNDIFFERENTIATED_RESOURCE_POOL_KEY = "undifferentiated_resource_pool"

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


def _discover_timetables(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        differentiated=True,
        pool_mapping: Optional[PoolMapping] = None,
) -> Dict[str, RCalendar]:
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
        start_time = event[log_ids.start_time]
        end_time = event[log_ids.end_time]

        calendar_factory.check_date_time(resource, activity, start_time)
        calendar_factory.check_date_time(resource, activity, end_time)

    calendars = calendar_factory.build_weekly_calendars(min_confidence, desired_support, min_participation)

    return calendars


def discover_undifferentiated(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
) -> Dict[str, RCalendar]:
    timetables = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, False)
    return timetables


def discover_per_resource_pool(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
) -> Tuple[Dict[str, RCalendar], PoolMapping]:
    analyzer = ResourcePoolDiscoverer(
        event_log[[log_ids.activity, log_ids.resource]],
        activity_key=log_ids.activity,
        resource_key=log_ids.resource)

    pool_mapping = __resource_pool_analyser_result_to_pool_mapping(analyzer.resource_table)

    # NOTE: ResourcePoolDiscoverer skips 'Start' and 'End' activities, so we need to add them manually to a system pool
    resource_names = event_log[log_ids.resource].unique()
    system_pool_name = 'System'
    for name in resource_names:
        if name not in pool_mapping:
            pool_mapping[name] = system_pool_name

    timetables_per_pool = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True, pool_mapping)

    # NOTE: Prosimos calendar discovery may return None for some resources, so we need either to change hyperparameters
    #  or put some default calendars for those resources. Below we do additional search for calendars for pools without
    #  any calendars.
    #
    # For None calendars, (a) take all those None pools and discover a new calendar for them,
    #  (b) If the calendar is still not found, then take all resources of the log and find one undifferentiated calendar
    #  (c) If the calendar is still None, assign 24/7

    # Helper dictionary to easily find resources without calendars
    pool_resources_by_pool_name = {}
    for pool_name in set(pool_mapping.values()):
        pool_resources_by_pool_name[pool_name] = set(
            filter(lambda resource_name, pool=pool_name: pool_mapping[resource_name] == pool,
                   pool_mapping)
        )

    # (a) Discovering a calendar for pools without any calendars as one single pool

    pools_without_timetables = set()
    for pool_name in pool_mapping.values():
        if pool_name not in timetables_per_pool or timetables_per_pool[pool_name] is None:
            pools_without_timetables.add(pool_name)

    if len(pools_without_timetables) > 0:
        # Creating one pool for all resource pool that don't have calendars

        resources_without_calendar = set()
        for resource_pool in pool_resources_by_pool_name.values():
            for resource in resource_pool:
                resources_without_calendar.add(resource)

        pool_name = 'missing_calendar'
        single_pool_mapping = {resource_name: pool_name for resource_name in resources_without_calendar}

        single_pool_mapping = PoolMapping(single_pool_mapping)

        # Single pool discovery

        timetable = _discover_timetables(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True,
            single_pool_mapping)

        if timetable[pool_name] is not None:
            for name in pools_without_timetables:
                timetables_per_pool[name] = timetable[pool_name]

    # (b) Discovering a calendar using all resources as one pool

    pools_without_timetables = set()
    for pool_name in pool_mapping.values():
        if pool_name not in timetables_per_pool or timetables_per_pool[pool_name] is None:
            pools_without_timetables.add(pool_name)

    if len(pools_without_timetables) > 0:
        # Creating one pool for all resources

        pool_name = 'missing_calendar'
        single_pool_mapping = PoolMapping({resource: pool_name for resource in event_log[log_ids.resource].unique()})

        timetable = _discover_timetables(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True,
            single_pool_mapping)

        if timetable[pool_name] is not None:
            for name in pools_without_timetables:
                timetables_per_pool[name] = timetable[pool_name]

    # (c) Assigning 24/7 to pools without any calendars

    pools_without_timetables = set()
    for pool_name in pool_mapping.values():
        if pool_name not in timetables_per_pool or timetables_per_pool[pool_name] is None:
            pools_without_timetables.add(pool_name)

    if len(pools_without_timetables) > 0:
        for pool_name in pools_without_timetables:
            timetables_per_pool[pool_name] = full_day_schedule(pool_name)

    return timetables_per_pool, pool_mapping


def discover_per_resource(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
) -> Dict[str, RCalendar]:  # resource_name -> RCalendar
    timetables_per_resource = _discover_timetables(
        event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True)

    # NOTE: Prosimos calendar discovery may return None for some resources, so we need either to change hyperparameters
    #  or put some default calendars for those resources. Below we do additional search for calendars for differentiated
    #  resources ignoring pools which are less likely to have a None calendar.
    #
    # For None calendars, (a) take all those None resources and discover a new calendar for them,
    #  (b) If the calendar is still not found, then take all resources of the log and find one undifferentiated calendar
    #  (c) If the calendar is still None, assign 24/7

    # (a) Discovering new calendar for resources without calendar using a pooled discovery

    resources_without_calendar = set(
        filter(lambda k: timetables_per_resource[k] is None, timetables_per_resource))

    if len(resources_without_calendar) > 0:
        log = event_log[event_log[log_ids.resource].isin(resources_without_calendar)]

        pool_name = 'missing_calendar'
        pool_mapping = PoolMapping({resource: pool_name for resource in resources_without_calendar})

        timetable = _discover_timetables(
            log, log_ids, granularity, min_confidence, desired_support, min_participation, True, pool_mapping)

        if timetable[pool_name] is not None:
            for name in resources_without_calendar:
                timetables_per_resource[name] = timetable[pool_name]

    # (b) Discovering a new calendar for the all resources as a single pool

    resources_without_calendar = set(
        filter(lambda k: timetables_per_resource[k] is None, timetables_per_resource))

    if len(resources_without_calendar) > 0:
        pool_name = 'missing_calendar'
        pool_mapping = PoolMapping({resource: pool_name for resource in event_log[log_ids.resource].unique()})

        timetable = _discover_timetables(
            event_log, log_ids, granularity, min_confidence, desired_support, min_participation, True, pool_mapping)

        if timetable[pool_name] is not None:
            for name in resources_without_calendar:
                timetables_per_resource[name] = timetable[pool_name]

    # (c) Assigning 24/7 to resources without calendar

    resources_without_calendar = set(
        filter(lambda k: timetables_per_resource[k] is None, timetables_per_resource))

    if len(resources_without_calendar) > 0:
        for resource in resources_without_calendar:
            timetables_per_resource[resource] = full_day_schedule(resource)

    return timetables_per_resource


def full_day_schedule(schedule_id: str = '24_7_CALENDAR') -> RCalendar:
    schedule = RCalendar(schedule_id)
    schedule.add_calendar_item(
        from_day="MONDAY",
        to_day="SUNDAY",
        begin_time="00:00:00.000",
        end_time="23:59:59.999",
    )
    return schedule


def working_hours_schedule(schedule_id: str = '9_5_CALENDAR') -> RCalendar:
    schedule = RCalendar(schedule_id)
    schedule.add_calendar_item(
        from_day="MONDAY",
        to_day="FRIDAY",
        begin_time="09:00:00.000",
        end_time="17:00:00.000",
    )
    return schedule

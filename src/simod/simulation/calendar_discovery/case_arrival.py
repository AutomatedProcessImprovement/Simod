import pandas as pd
from bpdfr_simulation_engine.resource_calendar import CalendarFactory

from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.parameters.calendars import Calendar, Timetable
from simod.utilities import nearest_divisor_for_granularity

UNDIFFERENTIATED_RESOURCE_POOL_KEY = "undifferentiated_resource_pool"


def _discover_undifferentiated(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4):
    calendar_factory = CalendarFactory(granularity)
    for (case_id, group) in event_log.groupby(by=log_ids.case):
        resource = UNDIFFERENTIATED_RESOURCE_POOL_KEY
        start_time = group[log_ids.start_time].min()
        end_time = group[log_ids.end_time].max()
        activity = case_id
        calendar_factory.check_date_time(resource, activity, start_time)
        calendar_factory.check_date_time(resource, activity, end_time)
    calendar_candidates = calendar_factory.build_weekly_calendars(min_confidence, desired_support, min_participation)
    calendar = {}
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            calendar[resource_id] = Timetable.from_list_of_dicts(calendar_candidates[resource_id].to_json())
    return calendar


def discover_undifferentiated(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4) -> Calendar:
    if 1440 % granularity != 0:
        granularity = nearest_divisor_for_granularity(granularity)

    timetables = _discover_undifferentiated(
        log, log_ids, granularity, min_confidence, desired_support, min_participation)
    timetables = timetables[UNDIFFERENTIATED_RESOURCE_POOL_KEY]

    calendar = Calendar(
        id='Undifferentiated_discovery',
        name='Undifferentiated_discovery',
        timetables=timetables
    )

    return calendar

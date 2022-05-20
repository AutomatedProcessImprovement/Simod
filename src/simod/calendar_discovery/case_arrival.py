import pandas as pd

from bpdfr_simulation_engine.resource_calendar import CalendarFactory

CASE_ID_KEY = 'case:concept:name'
START_TIMESTAMP_KEY = "start_timestamp"
END_TIMESTAMP_KEY = "time:timestamp"
UNDIFFERENTIATED_RESOURCE_POOL_KEY = "undifferentiated_resource_pool"


def discover(
        event_log: pd.DataFrame,
        granularity=60,
        min_confidence=0.1,
        desired_support=0.7,
        min_participation=0.4,
        columns_mapping: dict = None):
    # handling custom Simod column names
    case_id_column = CASE_ID_KEY
    end_time_column = END_TIMESTAMP_KEY
    if columns_mapping:
        case_id_column = columns_mapping[CASE_ID_KEY]
        end_time_column = columns_mapping[END_TIMESTAMP_KEY]

    calendar_factory = CalendarFactory(granularity)
    for (case_id, group) in event_log.groupby(by=[case_id_column]):
        resource = UNDIFFERENTIATED_RESOURCE_POOL_KEY
        start_time = group[START_TIMESTAMP_KEY].min()
        end_time = group[end_time_column].max()
        activity = case_id
        calendar_factory.check_date_time(resource, activity, start_time)
        calendar_factory.check_date_time(resource, activity, end_time)
    calendar_candidates = calendar_factory.build_weekly_calendars(min_confidence, desired_support, min_participation)
    calendar = {}
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            calendar[resource_id] = calendar_candidates[resource_id].to_json()
    return calendar

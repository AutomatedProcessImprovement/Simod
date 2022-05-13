import os
import uuid
from pathlib import Path

import pandas as pd

from bpdfr_simulation_engine.resource_calendar import CalendarFactory
from pm4py_wrapper.wrapper import convert_xes_to_csv


def test_calendar_module(entry_point):
    log_path = entry_point / 'PurchasingExample.xes'
    log_path_csv = log_path.with_stem(str(uuid.uuid4())).with_suffix('.csv')
    convert_xes_to_csv(log_path, log_path_csv)

    df = pd.read_csv(log_path_csv)
    log_path_csv.unlink()
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    calendar_factory = CalendarFactory(15)

    for (_, row) in df.iterrows():
        resource = row['org:resource']
        activity = row['concept:name']
        start_timestamp = row['start_timestamp'].to_pydatetime()
        end_timestamp = row['time:timestamp'].to_pydatetime()
        calendar_factory.check_date_time(resource, activity, start_timestamp)
        calendar_factory.check_date_time(resource, activity, end_timestamp)

    calendar_candidates = calendar_factory.build_weekly_calendars(0.1, 0.7, 0.4)

    calendar = {}
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            calendar[resource_id] = calendar_candidates[resource_id].to_json()

    assert len(calendar) > 0
    assert 'Kim Passa' in calendar



from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.util import interval_lifecycle

XES_RESOURCE_TAG = 'org:resource'
XES_CASE_ID_TAG = 'case:concept:name'
XES_ACTIVITY_TAG = 'concept:name'
XES_TIMESTAMP_TAG = 'time:timestamp'
XES_LIFECYCLE_TAG = 'lifecycle:transition'


@dataclass
class AuxiliaryLogRecord:
    event_id: int
    adjusted_duration_ms: int
    event_data: pd.Series
    duration_s: Optional[float] = None


def _make_auxiliary_log(log_path: Path) -> List[AuxiliaryLogRecord]:
    log = pm4py.read_xes(str(log_path))
    log_df = converter.apply(log, variant=converter.Variants.TO_DATA_FRAME)

    log_interval = interval_lifecycle.to_interval(log)
    log_interval_df = converter.apply(log_interval, variant=converter.Variants.TO_DATA_FRAME)

    # getting the necessary attributes
    # data = log_df[[XES_RESOURCE_TAG, XES_ACTIVITY_TAG, XES_TIMESTAMP_TAG, XES_LIFECYCLE_TAG]]
    data = log_df.sort_values(by=[XES_TIMESTAMP_TAG])

    # making an auxiliary log with sweep line algorithm
    active_set = {}
    previous_time_ms: int = 0
    aux_log: List[AuxiliaryLogRecord] = []
    for i, event in data.iterrows():
        if event[XES_LIFECYCLE_TAG] == 'start':
            event_id = data.index[i]  # using row index number as an event ID
            active_set[event_id] = event
        else:
            for e_id in active_set:
                active_set_event = active_set[e_id]
                if active_set_event is None:
                    continue
                # conversion to milliseconds to reduce the rounding error
                current_time_ms = active_set_event[XES_TIMESTAMP_TAG].timestamp() * 1000
                adjusted_duration = int(current_time_ms - previous_time_ms / len(active_set))
                record = AuxiliaryLogRecord(event_id=e_id, adjusted_duration_ms=adjusted_duration,
                                            event_data=data.loc[e_id],
                                            duration_s=_find_duration_by(active_set_event[XES_ACTIVITY_TAG],
                                                                         active_set_event[XES_RESOURCE_TAG],
                                                                         active_set_event[XES_CASE_ID_TAG],
                                                                         active_set_event[XES_TIMESTAMP_TAG],
                                                                         log_interval_df))
                aux_log.append(record)
                # we set to None instead of removing to avoid the exception "dictionary changed size during iteration"
                active_set[e_id] = None
        previous_time_ms = int(event[XES_TIMESTAMP_TAG].timestamp() * 1000)

    return aux_log


def _make_coalesced_log(records: List[AuxiliaryLogRecord]):
    log = pd.DataFrame()
    for event_id, group in groupby(records, lambda record: record.event_id):
        group_duration_ms = 0
        for record in group:
            group_duration_ms += record.adjusted_duration_ms
        group_duration_s = int(group_duration_ms / 1000.0)
        # event = record.event_data
        # event[XES_TIMESTAMP_TAG]
    pass


def _find_duration_by(activity: str, resource: str, case_id: str, start_timestamp: pd.Timestamp, data: pd.DataFrame) -> float:
    result = data[
        (data[XES_ACTIVITY_TAG] == activity) &
        (data[XES_RESOURCE_TAG] == resource) &
        (data[XES_CASE_ID_TAG] == case_id) &
        (data['start_timestamp'] == start_timestamp)
    ]
    if len(result) > 1:
        raise Exception('There should be only one event with such attributes')
    duration = result.iloc[0]['@@duration']
    return duration

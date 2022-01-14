from dataclasses import dataclass
from enum import Enum, auto
from itertools import groupby
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.util import interval_lifecycle

_XES_TIMESTAMP_TAG = 'time:timestamp'
_XES_RESOURCE_TAG = 'org:resource'
_PM4PY_START_TIMESTAMP_TAG = 'start_timestamp'


class _LifecycleType(Enum):
    START = auto()
    END = auto()

    def __str__(self):
        if self == _LifecycleType.START:
            return 'START'
        elif self == _LifecycleType.END:
            return 'END'
        raise ValueError(f'Unknown LifecycleType {str(self)}')


@dataclass
class _CustomLogRecord:
    event_id: int
    timestamp: pd.Timestamp
    lifecycle_type: _LifecycleType
    resource: str


@dataclass
class _AuxiliaryLogRecord:
    event_id: int
    timestamp: pd.Timestamp
    adjusted_duration_s: float


def adjust_durations(log_path: Path, verbose=False) -> pd.DataFrame:
    """Changes end timestamps for multitasking events without changing the overall resource utilization."""
    log = pm4py.read_xes(str(log_path))
    log_interval = interval_lifecycle.to_interval(log)
    log_interval_df = converter.apply(log_interval, variant=converter.Variants.TO_DATA_FRAME)
    if verbose:
        print('\nProcess: ', log_interval_df[['concept:name', 'org:resource', 'start_timestamp', 'time:timestamp']])
        utilization_before = _resource_utilization(log_interval_df)

    # apply sweep line for each resource
    resources = log_interval_df[_XES_RESOURCE_TAG].unique()
    for resource in resources:
        resource_events = log_interval_df[log_interval_df[_XES_RESOURCE_TAG] == resource]
        data = _make_custom_records(resource_events, log_interval_df)
        aux_log = _make_auxiliary_log(data)
        _update_end_timestamps(aux_log, log_interval_df)

    if verbose:
        utilization_after = _resource_utilization(log_interval_df)
        # print('Adjusted durations: ', list(filter(lambda record: record.adjusted_duration_s != 0, aux_log)))
        print('Resource utilization before', utilization_before)
        print('Resource utilization after', utilization_after)
        print('Utilization before equals the one after: ', utilization_before == utilization_after)
        print('New process: ', log_interval_df[['concept:name', 'org:resource', 'start_timestamp', 'time:timestamp']])
    return log_interval_df


def _make_auxiliary_log(data: List[_CustomLogRecord]) -> List[_AuxiliaryLogRecord]:
    """Adjusts duration for multitasking resources."""
    active_set: Dict[int, Optional[_CustomLogRecord]] = {}
    previous_time_s: float = 0
    aux_log: List[_AuxiliaryLogRecord] = []
    data = sorted(data, key=lambda item: item.timestamp)
    for record in data:
        for e_id in active_set:
            active_set_event = active_set[e_id]
            if active_set_event is None:
                continue
            current_time_s = record.timestamp.timestamp()
            adjusted_duration = (current_time_s - previous_time_s) / _active_set_len(active_set)
            aux_record = _AuxiliaryLogRecord(event_id=e_id,
                                             timestamp=active_set_event.timestamp,
                                             adjusted_duration_s=adjusted_duration)
            aux_log.append(aux_record)

        if record.lifecycle_type is _LifecycleType.START:
            active_set[record.event_id] = record
        else:
            # we set to None instead of removing to avoid the exception "dictionary changed size during iteration"
            active_set[record.event_id] = None

        previous_time_s = record.timestamp.timestamp()
    return aux_log


def _make_custom_records(resource_events: pd.DataFrame, log: pd.DataFrame):
    """Prepares records for the Sweep Line algorithm."""
    data: List[_CustomLogRecord] = []
    for i, event in resource_events.iterrows():
        start_item = _CustomLogRecord(event_id=log.index[i],
                                      timestamp=event[_PM4PY_START_TIMESTAMP_TAG],
                                      lifecycle_type=_LifecycleType.START,
                                      resource=event[_XES_RESOURCE_TAG])
        end_item = _CustomLogRecord(event_id=log.index[i],
                                    timestamp=event[_XES_TIMESTAMP_TAG],
                                    lifecycle_type=_LifecycleType.END,
                                    resource=event[_XES_RESOURCE_TAG])
        data.append(start_item)
        data.append(end_item)
    return data


def _active_set_len(active_set: dict) -> int:
    """Active set contains None values which we need to avoid in counting the amount of active items in the set."""
    length = 0
    for k in active_set:
        if active_set[k] is not None:
            length += 1
    return length


def _update_end_timestamps(records: List[_AuxiliaryLogRecord], log_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies end timestamp according to the adjusted durations."""
    for event_id, group in groupby(records, lambda record: record.event_id):
        duration = sum(map(lambda record: record.adjusted_duration_s, group))
        log_df.at[event_id, _XES_TIMESTAMP_TAG] = \
            log_df.loc[event_id][_PM4PY_START_TIMESTAMP_TAG] + pd.Timedelta(duration, unit='s')
    return log_df


def _resource_utilization(log: pd.DataFrame) -> Dict[str, float]:
    """Calculates resource utilization for each resource in the log."""
    resources = log[_XES_RESOURCE_TAG].unique()
    utilization = {}
    for resource in resources:
        events = log[log[_XES_RESOURCE_TAG] == resource]
        max_end = events[_XES_TIMESTAMP_TAG].max()
        min_start = events[_PM4PY_START_TIMESTAMP_TAG].min()
        end_timestamps = events[_XES_TIMESTAMP_TAG]
        start_timestamps = events[_PM4PY_START_TIMESTAMP_TAG]
        result = ((end_timestamps - start_timestamps) / (max_end - min_start)).sum()
        utilization[resource] = result
    return utilization


if __name__ == '__main__':
    import sys
    log_path = Path(sys.argv[1])
    _ = adjust_durations(log_path)

import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from itertools import groupby
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from simod.event_log import convert_df_to_xes, reformat_timestamps

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


def adjust_durations(log: pd.DataFrame, output_path: Optional[Path] = None, verbose=False,
                     is_concurrent=False, max_workers=multiprocessing.cpu_count()) -> pd.DataFrame:
    """Changes end timestamps for multitasking events without changing the overall resource utilization."""
    if verbose:
        metrics_before = _resource_metrics(log)

    # apply sweep line for each resource
    resources = log[_XES_RESOURCE_TAG].unique()
    if not is_concurrent:
        # sequential processing
        for resource in tqdm(resources, desc='processing resources'):
            _adjust_duration_for_resource(log, resource)
    else:
        # concurrent processing
        aux_log = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = (executor.submit(_make_aux_log, log, resource)
                       for resource in tqdm(resources, desc='submitting resources for concurrent processing'))
            for future in tqdm(concurrent.futures.as_completed(futures), desc='waiting for completion'):
                aux_records = future.result()
                aux_log.extend(aux_records)
        _update_end_timestamps(aux_log, log)

    if verbose:
        metrics = _resource_metrics(log)
        print('Utilization before equals the one after: ', metrics_before['utilization'] == metrics['utilization'])
        print("Resource events equal:", metrics_before['number_of_events'] == metrics['number_of_events'])

    if output_path is not None:
        log = log.drop(columns=[
            '@@startevent_org:resource',
            '@@startevent_concept:name',
            '@@startevent_Activity',
            '@@startevent_Resource',
            '@@duration',
            '@@custom_lif_id',
            '@@origin_ev_idx'],
            errors='ignore')

        convert_df_to_xes(log, output_path)

        reformat_timestamps(output_path, output_path)

    return log


def _adjust_duration_for_resource(log_interval_df, resource):
    resource_events = log_interval_df[log_interval_df[_XES_RESOURCE_TAG] == resource]
    data = _make_custom_records(resource_events, log_interval_df)
    aux_log = _make_auxiliary_log(data)
    _update_end_timestamps(aux_log, log_interval_df)


def _make_aux_log(log_interval_df, resource):
    resource_events = log_interval_df[log_interval_df[_XES_RESOURCE_TAG] == resource]
    data = _make_custom_records(resource_events, log_interval_df)
    return _make_auxiliary_log(data)


def _make_custom_records(resource_events: pd.DataFrame, log: pd.DataFrame):
    """Prepares records for the Sweep Line algorithm."""
    data: List[_CustomLogRecord] = []
    for i, event in resource_events.iterrows():
        start_timestamp = event[_PM4PY_START_TIMESTAMP_TAG]
        end_timestamp = event[_XES_TIMESTAMP_TAG]
        resource = event[_XES_RESOURCE_TAG]
        event_id = log.index[i]
        if start_timestamp == end_timestamp:  # filter out instant events
            continue
        start_item = _CustomLogRecord(event_id=event_id,
                                      timestamp=start_timestamp,
                                      lifecycle_type=_LifecycleType.START,
                                      resource=resource)
        end_item = _CustomLogRecord(event_id=event_id,
                                    timestamp=end_timestamp,
                                    lifecycle_type=_LifecycleType.END,
                                    resource=resource)
        data.extend([start_item, end_item])
    return data


def _make_auxiliary_log(data: List[_CustomLogRecord]) -> List[_AuxiliaryLogRecord]:
    """Adjusts duration for multitasking resources."""
    active_set: Dict[int, Optional[_CustomLogRecord]] = {}
    previous_time_s: float = 0
    aux_log: List[_AuxiliaryLogRecord] = []
    data = sorted(data, key=lambda item: item.timestamp)
    for record in data:
        current_time_s = record.timestamp.timestamp()
        adjusted_duration = 0
        active_set_len = len(active_set)
        if active_set_len > 0:
            adjusted_duration = (current_time_s - previous_time_s) / active_set_len
        aux_log.extend(_AuxiliaryLogRecord(event_id=e_id,
                                           timestamp=active_set[e_id].timestamp,
                                           adjusted_duration_s=adjusted_duration)
                       for e_id in active_set)
        previous_time_s = record.timestamp.timestamp()
        if record.lifecycle_type is _LifecycleType.START:
            active_set[record.event_id] = record
        else:
            del active_set[record.event_id]
    return aux_log


def _update_end_timestamps(records: List[_AuxiliaryLogRecord], log_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies end timestamp according to the adjusted durations."""
    records = sorted(records, key=lambda record: record.event_id)  # groupby below works only on sorted data
    for event_id, group in groupby(records, lambda record: record.event_id):
        duration = sum(map(lambda record: record.adjusted_duration_s, group))
        log_df.at[event_id, _XES_TIMESTAMP_TAG] = \
            log_df.loc[event_id][_PM4PY_START_TIMESTAMP_TAG] + pd.Timedelta(duration, unit='s')
    return log_df


def _resource_metrics(log: pd.DataFrame) -> dict:
    """Calculates resource utilization for each resource in the log."""
    resources = log[_XES_RESOURCE_TAG].unique()
    utilization = {}
    number_of_events = {}
    for resource in resources:
        events = log[log[_XES_RESOURCE_TAG] == resource]
        number_of_events[resource] = len(events)
        max_end = events[_XES_TIMESTAMP_TAG].max()
        min_start = events[_PM4PY_START_TIMESTAMP_TAG].min()
        end_timestamps = events[_XES_TIMESTAMP_TAG]
        start_timestamps = events[_PM4PY_START_TIMESTAMP_TAG]
        result = ((end_timestamps - start_timestamps) / (max_end - min_start)).sum()
        utilization[resource] = result
    return {'utilization': utilization, 'number_of_events': number_of_events}


if __name__ == '__main__':
    import sys

    log_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    adjust_durations(log_path, output_path, verbose=True)

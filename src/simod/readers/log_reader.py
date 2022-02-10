import itertools
import os
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from typing import Optional

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import interval_lifecycle

DEFAULT_CSV_COLUMNS = {
    'concept:name': 'task',
    'case:concept:name': 'caseid',
    'lifecycle:transition': 'event_type',
    'org:resource': 'user',
    'time:timestamp': 'end_timestamp'
}

QBP_CSV_COLUMNS = {
    'resource': 'user'
}

DEFAULT_XES_COLUMNS = {
    'Start Timestamp': 'start_timestamp',
    'Complete Timestamp': 'end_timestamp',
    'resource': 'user'
}

DEFAULT_COLUMNS = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


class LogReader:
    log_path: Path
    log_path_xes: Path
    log: pd.DataFrame
    data: list  # TODO: we need to get rid of list and use DataFrame everywhere
    _time_format: str
    _column_names: dict
    _column_filter: Optional[list] = None

    def __init__(self,
                 log_path: Path,
                 column_names: dict = DEFAULT_CSV_COLUMNS,
                 column_filter: Optional[list] = DEFAULT_COLUMNS,
                 time_format: str = TIME_FORMAT,
                 load: bool = True,
                 log: Optional[pd.DataFrame] = None):
        if not log_path.exists():
            raise FileNotFoundError

        _, ext = os.path.splitext(log_path)
        if ext != '.csv':
            # NOTE: XES is needed for CalenderImp Java dependency which will be removed later
            self.log_path_xes = log_path
            self.log_path = log_path.with_suffix('.csv')
            convert_xes_to_csv(log_path, self.log_path)
        else:
            self.log_path = log_path

            # NOTE: we assume that XES is located at the same path
            self.log_path_xes = log_path.with_suffix('.xes')  # TODO: should we convert CSV to XES if XES isn't provided

        if load:
            if log is None:
                df = pd.read_csv(self.log_path)
            else:
                df = log
            df = df.rename(columns=column_names)
            df = df.astype({'caseid': object})
            df = df[(df.task != 'Start') & (df.task != 'End')].reset_index(drop=True)
            if column_filter is not None:
                df = df[column_filter]
            df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format=time_format, utc=True)
            df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format=time_format, utc=True)
            self.log = df

            self.data = df.to_dict('records')
            self.data = self._append_csv_start_end_entries(self.data)

        self._time_format = time_format
        self._column_names = column_names
        self._column_filter = column_filter

    @staticmethod
    def copy_without_data(log: 'LogReader') -> 'LogReader':
        """Copies LogReader without copying underlying data."""
        reader = LogReader(log_path=log.log_path, load=False)
        reader._time_format = log._time_format
        reader._column_names = log._column_names
        return reader

    @staticmethod
    def _append_csv_start_end_entries(data: list) -> list:
        """Adds START and END activities at the beginning and end of each trace."""
        end_start_times = dict()
        log = pd.DataFrame(data)
        for case, group in log.groupby('caseid'):
            end_start_times[(case, 'Start')] = group.start_timestamp.min() - timedelta(microseconds=1)
            end_start_times[(case, 'End')] = group.end_timestamp.max() + timedelta(microseconds=1)
        new_data = []
        data = sorted(data, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = end_start_times[(key, new_event)]
                temp_event['start_timestamp'] = end_start_times[(key, new_event)]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        return new_data

    def set_data(self, data: list):
        self.data = data

    def get_traces(self):
        """Returns the data split by caseid and ordered by start_timestamp."""
        cases = list(set([x['caseid'] for x in self.data]))
        traces = []
        for case in cases:
            order_key = 'start_timestamp'
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter(order_key))
            traces.append(trace)
        return traces


def convert_xes_to_csv(xes_path: Path, output_path: Path):
    log = pm4py.read_xes(str(xes_path))
    log_interval = interval_lifecycle.to_interval(log)
    df = log_converter.apply(log_interval, variant=log_converter.Variants.TO_DATA_FRAME)
    df.to_csv(output_path, index=False)


def convert_xes_to_csv_if_needed(log_path: Path, output_path: Optional[Path] = None) -> Path:
    _, ext = os.path.splitext(log_path)
    if ext != '.csv':
        if output_path:
            log_path_csv = output_path
        else:
            log_path_csv = log_path.with_suffix('.csv')
        convert_xes_to_csv(log_path, log_path_csv)
        return log_path_csv
    else:
        return log_path


def read(log_path: Path) -> pd.DataFrame:
    log_path_csv = convert_xes_to_csv_if_needed(log_path)
    log = pd.read_csv(log_path_csv)
    log['start_timestamp'] = pd.to_datetime(log['start_timestamp'], utc=True)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)
    return log

import copy
import itertools
import os
from _operator import itemgetter
from dataclasses import dataclass
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from typing import Union, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pendulum

from simod.cli_formatter import print_step

DEFAULT_XES_COLUMNS = {
    'concept:name': 'task',
    'case:concept:name': 'caseid',
    'lifecycle:transition': 'event_type',
    'org:resource': 'user',
    'time:timestamp': 'end_timestamp'
}

QBP_CSV_COLUMNS = {
    'resource': 'user'
}

DEFAULT_FILTER = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


@dataclass
class EventLogIDs:
    """Column mapping between the event log."""
    case: str = 'case_id'
    activity: str = 'activity'
    resource: str = 'resource'
    start_time: str = 'start_timestamp'
    end_time: str = 'end_timestamp'
    enabled_time: str = 'enabled_timestamp'
    role: str = 'role'


class LogReader:
    log_path: Path
    # log_path_xes: Path
    df: pd.DataFrame
    data: list  # TODO: we need to get rid of list and use DataFrame everywhere
    _time_format: str
    _column_names: dict
    _column_filter: Optional[list] = None

    def __init__(self,
                 log_path: Path,
                 column_names: dict = DEFAULT_XES_COLUMNS,
                 column_filter: Optional[list] = DEFAULT_FILTER,
                 time_format: str = TIME_FORMAT,
                 load: bool = True,
                 log: Optional[pd.DataFrame] = None):
        if isinstance(log_path, str):
            log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f'Log file {log_path} does not exist.')

        self.df = log
        self.log_path = log_path
        # _, ext = os.path.splitext(log_path)
        # if ext != '.csv':
        #     # NOTE: XES is needed for CalenderImp Java dependency which will be removed later
        #     self.log_path_xes = log_path
        #     self.log_path = log_path.with_suffix('.csv')
        #     convert_xes_to_csv(log_path, self.log_path)
        # else:
        #     self.log_path = log_path
        #
        #     # NOTE: we assume that XES is located at the same path
        #     self.log_path_xes = log_path.with_suffix('.xes')
        # TODO: should we convert CSV to XES if XES isn't provided

        if load:
            self._read_log(log, column_filter, column_names, time_format)

        self._time_format = time_format
        self._column_names = column_names
        self._column_filter = column_filter

    def _read_log(self, log: Optional[pd.DataFrame], column_filter: list, column_names: dict, time_format: str):
        if log is None:
            df, log_path_csv = read(self.log_path)
        else:
            df = log

        assert len(df) > 0, 'Log is empty'

        # renaming for internal use
        df.rename(columns=column_names, inplace=True)

        # type conversion
        # df = df.astype({'caseid': object})
        convert_timestamps(df)

        # filtering out Start and End fake events
        df = df[(df.task != 'Start') & (df.task != 'End')].reset_index(drop=True)

        # filtering columns
        if column_filter is not None:
            df = df[column_filter]

        self.data = df.to_dict('records')
        self.data = self._append_csv_start_end_entries(self.data)
        self.df = pd.DataFrame(self.data)  # TODO: can we these log manipulations clearer?

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
        self.df = pd.DataFrame(self.data)

    def get_traces(self):
        """Returns the data split by caseid and ordered by start_timestamp."""
        cases = list(set([x['caseid'] for x in self.data]))
        traces = []
        for case in cases:
            order_key = 'start_timestamp'
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def get_traces_df(self, include_start_end_events: bool = False) -> pd.DataFrame:
        if include_start_end_events:
            return self.df
        return self.df[(self.df.task != 'Start') & (self.df.task != 'End')].reset_index(drop=True)

    def split_timeline(self, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split an event log dataframe by time to perform split-validation. preferred method time splitting removing
        incomplete traces. If the testing set is smaller than the 10% of the log size the second method is sort by traces
        start and split taking the whole traces no matter if they are contained in the timeframe or not
        """
        # Split log data
        splitter = LogSplitter(self.df)
        partition1, partition2 = splitter.split_log('timeline_contained', size)
        total_events = len(self.df)

        # Check size and change time splitting method if necesary
        if len(partition2) < int(total_events * 0.1):
            partition1, partition2 = splitter.split_log('timeline_trace', size)

        # Set splits
        partition1 = pd.DataFrame(partition1)
        partition2 = pd.DataFrame(partition2)
        return partition1, partition2


def write_xes(log: Union[LogReader, pd.DataFrame, list], output_path: Path):
    log_df: pd.DataFrame

    if isinstance(log, pd.DataFrame):
        log_df = log
    elif isinstance(log, LogReader):
        log_df = pd.DataFrame(log.data)
    elif isinstance(log, list):
        log_df = pd.DataFrame(log)
    else:
        raise Exception(f'Unimplemented type for {type(log)}')

    log_df.rename(columns={
        'task': 'concept:name',
        'caseid': 'case:concept:name',
        'event_type': 'lifecycle:transition',
        'user': 'org:resource',
        'end_timestamp': 'time:timestamp'
    }, inplace=True)

    log_df.drop(columns=['@@startevent_concept:name',
                         '@@startevent_org:resource',
                         '@@startevent_Activity',
                         '@@startevent_Resource',
                         '@@duration',
                         'case:variant',
                         'case:variant-index',
                         'case:creator',
                         'Activity',
                         'Resource',
                         'elementId',
                         'processId',
                         'resourceId',
                         'resourceCost',
                         '@@startevent_element',
                         '@@startevent_elementId',
                         '@@startevent_process',
                         '@@startevent_processId',
                         '@@startevent_resourceId',
                         'etype'],
                inplace=True,
                errors='ignore')

    log_df.fillna('UNDEFINED', inplace=True)

    convert_df_to_xes(log_df, output_path)


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


def read(log_path: Path) -> Tuple[pd.DataFrame, Path]:
    log_path_csv = convert_xes_to_csv_if_needed(log_path)
    log = pd.read_csv(log_path_csv)
    convert_timestamps(log)
    return log, log_path_csv


def convert_timestamps(log: pd.DataFrame):
    time_columns = ['start_timestamp', 'enabled_timestamp', 'end_timestamp', 'time:timestamp']
    for name in time_columns:
        if name in log.columns:
            log[name] = pd.to_datetime(log[name], utc=True)


def convert_xes_to_csv(xes_path: Path, csv_path: Path):
    args = ['pm4py_wrapper', '-i', str(xes_path), '-o', str(csv_path.parent), 'xes-to-csv']
    print_step(f'Executing shell command [convert_xes_to_csv]: {args}')
    os.system(' '.join(args))


def convert_df_to_xes(df: pd.DataFrame, output_path: Path):
    xes_datetime_format = 'YYYY-MM-DDTHH:mm:ss.SSSZ'
    df['start_timestamp'] = df['start_timestamp'].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    df['time:timestamp'] = df['time:timestamp'].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    df.to_csv(output_path, index=False)
    args = ['pm4py_wrapper', '-i', str(output_path), '-o', str(output_path.parent), 'csv-to-xes']
    print_step(f'Executing shell command [convert_df_to_xes]: {args}')
    os.system(' '.join(args))


def reformat_timestamps(xes_path: Path, output_path: Path):
    """Converts timestamps in XES to a format suitable for the Simod's calendar Java dependency."""
    ns = 'http://www.xes-standard.org/'
    date_tag = f'{{{ns}}}date'

    ET.register_namespace('', ns)
    tree = ET.parse(xes_path)
    root = tree.getroot()
    xpaths = [
        ".//*[@key='time:timestamp']",
        ".//*[@key='start_timestamp']"
    ]
    for xpath in xpaths:
        for element in root.iterfind(xpath):
            try:
                timestamp = pd.to_datetime(element.get('value'), format='%Y-%m-%d %H:%M:%S')
                value = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')
                element.set('value', value)
                element.tag = date_tag
            except ValueError:
                continue
    tree.write(output_path)


class LogSplitter:
    def __init__(self, log: pd.DataFrame):
        self.log = log
        self._sort_log()

    def split_log(self, method: str, size: float, one_timestamp: bool = False):
        splitter = self._get_splitter(method)
        return splitter(size, one_timestamp)

    def _get_splitter(self, method):
        if method == 'timeline_contained':
            return self._timeline_contained
        elif method == 'timeline_trace':
            return self._timeline_trace
        elif method == 'random':
            return self._random
        else:
            raise ValueError(method)

    def _timeline_contained(self, size: float, one_timestamp: bool):
        # log = self.log.data.to_dict('records')
        num_events = int(np.round(len(self.log) * (1 - size)))

        df_train = self.log.iloc[num_events:]
        df_test = self.log.iloc[:num_events]

        # Incomplete final traces
        df_train = df_train.sort_values(by=['caseid', 'pos_trace'],
                                        ascending=True)
        inc_traces = pd.DataFrame(df_train.groupby('caseid')
                                  .last()
                                  .reset_index())
        inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
        inc_traces = inc_traces['caseid'].to_list()

        # Drop incomplete traces
        df_test = df_test[~df_test.caseid.isin(inc_traces)]
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])

        df_train = df_train[~df_train.caseid.isin(inc_traces)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        df_test = (df_test
                   .sort_values(key, ascending=True)
                   .reset_index(drop=True).to_dict('records'))
        df_train = (df_train
                    .sort_values(key, ascending=True)
                    .reset_index(drop=True).to_dict('records'))
        return df_train, df_test

    def _timeline_trace(self, size: float, one_timestamp: bool):
        # log = self.log.data.to_dict('records')
        cases = self.log[self.log.pos_trace == 1]
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        cases = cases.sort_values(key, ascending=False)
        cases = cases.caseid.to_list()
        num_test_cases = int(np.round(len(cases) * (1 - size)))
        test_cases = cases[:num_test_cases]
        train_cases = cases[num_test_cases:]
        df_train = self.log[self.log.caseid.isin(train_cases)]
        df_test = self.log[self.log.caseid.isin(test_cases)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])
        return df_train, df_test

    def _random(self, size: float, one_timestamp: bool):
        cases = list(self.log.caseid.unique())
        sample_sz = int(np.ceil(len(cases) * size))
        scases = list(map(lambda i: cases[i], np.random.randint(0, len(cases), sample_sz)))
        df_train = self.log[self.log.caseid.isin(scases)]
        df_test = self.log[~self.log.caseid.isin(scases)]
        return df_train, df_test

    def _sort_log(self):
        log = copy.deepcopy(self.log)
        log = sorted(log.to_dict('records'), key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_dict(log)
        log.sort_values(by='end_timestamp', ascending=False, inplace=True)
        self.log = log

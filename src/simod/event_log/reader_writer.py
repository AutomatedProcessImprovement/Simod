import itertools
from _operator import itemgetter
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from simod.event_log.splitter import LogSplitter
from simod.event_log.utilities import read, convert_timestamps, convert_df_to_xes

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


class LogReaderWriter:
    log_path: Path
    # log_path_xes: Path
    df: pd.DataFrame
    data: list  # TODO: remove the list and use DataFrame everywhere
    _time_format: str
    _column_names: dict  # TODO: use EventLogIDs
    _column_filter: Optional[list] = None

    def __init__(self,
                 log_path: Path,
                 column_names: dict = DEFAULT_XES_COLUMNS,  # TODO: replace with EventLogIDs
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
    def copy_without_data(log: 'LogReaderWriter') -> 'LogReaderWriter':
        """Copies LogReader without copying underlying data."""
        reader = LogReaderWriter(log_path=log.log_path, load=False)
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

    def write_xes(self, output_path: Path):
        log_df = pd.DataFrame(self.data)

        # TODO: use EventLogIDs
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

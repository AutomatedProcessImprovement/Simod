import itertools
from _operator import itemgetter
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS
from simod.event_log.splitter import LogSplitter
from simod.event_log.utilities import read, convert_timestamps, convert_df_to_xes

DEFAULT_XES_COLUMNS = {
    'concept:name': 'task',
    'case:concept:name': 'caseid',
    'lifecycle:transition': 'event_type',
    'org:resource': 'user',
    'time:timestamp': 'end_timestamp'
}


class LogReaderWriter:
    log_path: Path
    df: pd.DataFrame
    data: list  # TODO: remove the list and use DataFrame everywhere
    _log_ids: EventLogIDs

    def __init__(self,
                 log_path: Path,
                 log_ids: EventLogIDs,
                 load: bool = True,
                 log: Optional[pd.DataFrame] = None):
        if isinstance(log_path, str):
            log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f'Log file {log_path} does not exist.')

        self.df = log
        self.log_path = log_path
        self._log_ids = log_ids

        if load:
            self._read_log(log)

    def _read_log(self, log: Optional[pd.DataFrame]):
        if log is None:
            df, log_path_csv = read(self.log_path, self._log_ids)
        else:
            df = log

        assert len(df) > 0, f'Log {self.log_path} is empty'

        # renaming for internal use
        # df.rename(columns=column_names, inplace=True)

        # type conversion
        # df = df.astype({'caseid': object})
        convert_timestamps(df, self._log_ids)

        # filtering out Start and End fake events
        df = df[(df[self._log_ids.activity] != 'Start') & (df[self._log_ids.activity] != 'End')].reset_index(drop=True)
        df = df[(df[self._log_ids.activity] != 'start') & (df[self._log_ids.activity] != 'end')].reset_index(drop=True)

        # filtering columns
        # if column_filter is not None:
        #     df = df[column_filter]
        # ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']
        df = df[[
            self._log_ids.case,
            self._log_ids.activity,
            self._log_ids.resource,
            self._log_ids.start_time,
            self._log_ids.end_time
        ]]

        self.data = df.to_dict('records')
        self.data = self._append_csv_start_end_entries(self.data)
        self.df = pd.DataFrame(self.data)  # TODO: can we these log manipulations clearer?

    @staticmethod
    def copy_without_data(log: 'LogReaderWriter', log_ids: EventLogIDs) -> 'LogReaderWriter':
        """Copies LogReader without copying underlying data."""
        reader = LogReaderWriter(log_path=log.log_path, log_ids=log_ids, load=False)
        return reader

    def _append_csv_start_end_entries(self, data: list) -> list:
        """Adds START and END activities at the beginning and end of each trace."""
        end_start_times = dict()
        log = pd.DataFrame(data)
        for case, group in log.groupby(self._log_ids.case):
            end_start_times[(case, 'Start')] = group[self._log_ids.start_time].min() - timedelta(microseconds=1)
            end_start_times[(case, 'End')] = group[self._log_ids.end_time].max() + timedelta(microseconds=1)
        new_data = []
        data = sorted(data, key=lambda x: x[self._log_ids.case])
        for key, group in itertools.groupby(data, key=lambda x: x[self._log_ids.case]):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = {
                    self._log_ids.case: trace[idx][self._log_ids.case],
                    self._log_ids.activity: new_event,
                    self._log_ids.resource: new_event,
                    self._log_ids.end_time: end_start_times[(key, new_event)],
                    self._log_ids.start_time: end_start_times[(key, new_event)],
                }
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
        cases = list(set([x[self._log_ids.case] for x in self.data]))
        traces = []
        for case in cases:
            order_key = self._log_ids.start_time
            trace = sorted(
                list(filter(lambda x: (x[self._log_ids.case] == case), self.data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def get_traces_df(self, include_start_end_events: bool = False) -> pd.DataFrame:
        if include_start_end_events:
            return self.df
        return self.df[
            (self.df[self._log_ids.activity] not in ('Start', 'start')) & (
                    self.df[self._log_ids.activity] not in ('End', 'end'))].reset_index(
            drop=True)

    def split_timeline(self, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split an event log dataframe by time to perform split-validation. preferred method time splitting removing
        incomplete traces. If the testing set is smaller than the 10% of the log size the second method is sort by traces
        start and split taking the whole traces no matter if they are contained in the timeframe or not
        """
        # Split log data
        splitter = LogSplitter(self.df, self._log_ids)
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

        log_df.rename(columns={
            self._log_ids.activity: 'concept:name',
            self._log_ids.case: 'case:concept:name',
            self._log_ids.resource: 'org:resource',
            self._log_ids.start_time: 'start_timestamp',
            self._log_ids.end_time: 'time:timestamp',
            'event_type': 'lifecycle:transition',
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

        convert_df_to_xes(log_df, STANDARD_COLUMNS, output_path)

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS
from simod.event_log.splitter import LogSplitter
from simod.event_log.utilities import read, convert_timestamps, convert_df_to_xes


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

        convert_timestamps(df, self._log_ids)

        df = df[[
            self._log_ids.case,
            self._log_ids.activity,
            self._log_ids.resource,
            self._log_ids.start_time,
            self._log_ids.end_time
        ]]

        self.data = df.to_dict('records')
        self.df = pd.DataFrame(self.data)  # TODO: can we these log manipulations clearer?

    @staticmethod
    def copy_without_data(log: 'LogReaderWriter', log_ids: EventLogIDs) -> 'LogReaderWriter':
        """Copies LogReader without copying underlying data."""
        reader = LogReaderWriter(log_path=log.log_path, log_ids=log_ids, load=False)
        return reader

    def set_data(self, data: list):
        self.data = data
        self.df = pd.DataFrame(self.data)

    def get_traces_df(self) -> pd.DataFrame:
        return self.df

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

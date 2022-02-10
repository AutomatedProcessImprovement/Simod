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
                 load: bool = True):
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
            self.log_path_xes = log_path.with_suffix('.xes')

        if load:
            df = pd.read_csv(self.log_path)
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
    df.to_csv(output_path)

#
# class LogReader:
#     """This class reads and parses the elements of a given event-log. Expected formats: .xes or .csv."""
#     data: list
#     _input: Union[Path, str]
#     _file_name: str
#     _file_extension: str
#     _raw_data: list
#     _verbose: bool
#     _filter_attributes: bool
#     _column_names: Dict[str, str]
#     _columns_filter: List[str]
#     _time_format: str
#
#     def __init__(self, input: Union[Path, str], settings: ReadOptions, verbose=True, load=True):
#         if isinstance(input, Path):
#             self._input = input.absolute().__str__()
#         else:
#             self._input = input
#         self._file_name, self._file_extension = self._define_log_type()
#         self._time_format = settings.timeformat
#         self._column_names = self._default_columns_map()
#         if self._file_extension == '.csv':  # TODO: for some reason we ignore ReadOptions defaults (or user provided columns) for XES here
#             self._column_names.update(settings.column_names)
#         self._columns_filter = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']
#         self._filter_attributes = settings.filter_d_attrib
#         self._verbose = verbose
#         self._raw_data = list()
#         self.data = list()
#         if load:
#             self._load_data_from_file()
#
#     @staticmethod
#     def copy_without_data(log: 'LogReader') -> 'LogReader':
#         default_options = ReadOptions(column_names=ReadOptions.column_names_default())
#         copy = LogReader(input=log._input, settings=default_options, load=False)
#         copy._time_format = log._time_format
#         copy._column_names = log._column_names
#         copy._filter_attributes = log._filter_attributes
#         copy._verbose = log._verbose
#         return copy
#
#     def get_traces(self):  # TODO: can we do it just with Pandas functions?
#         """Returns the data split by caseid and ordered by start_timestamp."""
#         cases = list(set([x['caseid'] for x in self.data]))
#         traces = list()
#         for case in cases:
#             order_key = 'start_timestamp'
#             trace = sorted(
#                 list(filter(lambda x: (x['caseid'] == case), self.data)),
#                 key=itemgetter(order_key))
#             traces.append(trace)
#         return traces
#
#     def set_data(self, data):
#         self.data = data
#
#     def _load_data_from_file(self):
#         if self._verbose:
#             print_step('Reading log traces')
#         if self._file_extension == '.xes':
#             self._get_xes_events_data()
#         elif self._file_extension == '.csv':
#             self._get_csv_events_data()
#
#     def _get_xes_events_data(self):
#         log = pm4py.read_xes(self._input)
#         try:
#             source = log.attributes['source']
#         except KeyError:
#             source = ''
#         log = interval_lifecycle.to_interval(log)
#
#         end_time_key = 'time:timestamp'
#         start_time_key = 'start_timestamp'
#         flattened_log = [{**event, 'caseid': trace.attributes['concept:name']} for trace in log for event in trace]
#         temp_data = pd.DataFrame(flattened_log)
#
#         # NOTE: Stripping zone information from the log
#         # TODO: is it applicable here: pm4py.objects.log.util.dataframe_utils.convert_timestamp_columns_in_df?
#         temp_data[end_time_key] = temp_data.apply(lambda x: x[end_time_key].strftime(self._time_format), axis=1)
#         self._convert_timestamps(temp_data, end_time_key, utc=False)
#         temp_data[start_time_key] = temp_data.apply(lambda x: x[start_time_key].strftime(self._time_format), axis=1)
#         self._convert_timestamps(temp_data, start_time_key, utc=False)
#
#         temp_data.rename(columns=self._column_names, inplace=True)
#         temp_data = temp_data[~temp_data.task.isin(['Start', 'End', 'start', 'end'])].reset_index(drop=True)
#
#         if source == 'com.qbpsimulator' and len(temp_data.iloc[0].elementId.split('_')) > 1:
#             temp_data['etype'] = temp_data.apply(lambda x: x.elementId.split('_')[0], axis=1)
#             temp_data = (temp_data[temp_data.etype == 'Task'].reset_index(drop=True))
#
#         self._raw_data = temp_data.to_dict('records')
#         temp_data.drop_duplicates(inplace=True)
#         self.data = temp_data.to_dict('records')
#         self._append_csv_start_end()
#
#     def _get_csv_events_data(self):
#         """Reads and parse all the events information from a csv file."""
#         log = pd.read_csv(self._input)
#         log = log.rename(columns=self._column_names)
#         log = log.astype({'caseid': object})
#         log = log[(log.task != 'Start') & (log.task != 'End')].reset_index(drop=True)
#         if self._filter_attributes:
#             log = log[self._columns_filter]
#         self._convert_timestamps(log, 'start_timestamp')
#         self._convert_timestamps(log, 'end_timestamp')
#         self.data = log.to_dict('records')
#         self._append_csv_start_end()
#
#     def _convert_timestamps(self, log, column_name, utc=True):
#         log[column_name] = pd.to_datetime(log[column_name], format=self._time_format, utc=utc)
#
#     def _default_columns_map(self):
#         if self._file_extension is None:
#             raise ValueError
#         if self._file_extension == '.xes':
#             return {
#                 'concept:name': 'task',
#                 'case:concept:name': 'caseid',
#                 'lifecycle:transition': 'event_type',
#                 'org:resource': 'user',
#                 'time:timestamp': 'end_timestamp'
#             }
#         elif self._file_extension == '.csv':
#             return {
#                 'Start Timestamp': 'start_timestamp',
#                 'Complete Timestamp': 'end_timestamp',
#                 'resource': 'user'
#             }
#
#     def _append_csv_start_end(self):
#         end_start_times = dict()
#         for case, group in pd.DataFrame(self.data).groupby('caseid'):
#             end_start_times[(case, 'Start')] = group.start_timestamp.min() - timedelta(microseconds=1)
#             end_start_times[(case, 'End')] = group.end_timestamp.max() + timedelta(microseconds=1)
#         new_data = list()
#         data = sorted(self.data, key=lambda x: x['caseid'])
#         for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
#             trace = list(group)
#             for new_event in ['Start', 'End']:
#                 idx = 0 if new_event == 'Start' else -1
#                 temp_event = dict()
#                 temp_event['caseid'] = trace[idx]['caseid']
#                 temp_event['task'] = new_event
#                 temp_event['user'] = new_event
#                 temp_event['end_timestamp'] = end_start_times[(key, new_event)]
#                 temp_event['start_timestamp'] = end_start_times[(key, new_event)]
#                 if new_event == 'Start':
#                     trace.insert(0, temp_event)
#                 else:
#                     trace.append(temp_event)
#             new_data.extend(trace)
#         self.data = new_data
#
#     def _define_log_type(self):
#         filename, file_extension = os.path.splitext(self._input)
#         if file_extension in ['.xes', '.csv', '.mxml']:
#             filename = filename + file_extension
#         elif file_extension == '.gz':
#             out_filename = filename
#             filename, file_extension = self._decompress_gzip(out_filename)
#         elif file_extension == '.zip':
#             filename, file_extension = self._decompress_zip(filename)
#         else:
#             raise IOError('file type not supported')
#         return filename, file_extension
#
#     def _decompress_gzip(self, out_filename):
#         in_file = gzip.open(self._input, 'rb')
#         out_file = open(out_filename, 'wb')
#         out_file.write(in_file.read())
#         in_file.close()
#         out_file.close()
#         _, file_extension = os.path.splitext(out_filename)
#         return out_filename, file_extension
#
#     def _decompress_zip(self, out_filename):
#         with zf.ZipFile(self._input, "r") as zip_ref:
#             zip_ref.extractall("../inputs/")
#         _, file_extension = os.path.splitext(out_filename)
#         return out_filename, file_extension

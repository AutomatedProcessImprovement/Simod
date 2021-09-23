import gzip
import itertools
import os
import zipfile as zf
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from typing import Union

import pandas as pd
import pm4py
from pm4py.objects.log.util import interval_lifecycle

from .. import support_utils as sup
from ..configuration import ReadOptions


class LogReaderOld(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    def __init__(self, input: Union[Path, str], settings: ReadOptions, verbose=True, load=True):
        if isinstance(input, Path):
            self.input = input.absolute().__str__()
        else:
            self.input = input
        self.file_name, self.file_extension = self.define_ftype()

        self.timeformat = settings.timeformat
        self.column_names = settings.column_names
        self.one_timestamp = settings.one_timestamp
        self.filter_d_attrib = settings.filter_d_attrib
        self.verbose = verbose

        self.data = list()
        self.raw_data = list()

        if load:
            self.load_data_from_file()

    def load_data_from_file(self):
        """
        reads all the data from the log depending
        the extension of the file
        """
        # TODO: esto se puede manejar mejor con un patron fabrica
        if self.file_extension == '.xes':
            self.get_xes_events_data()
        elif self.file_extension == '.csv':
            self.get_csv_events_data()

    # =============================================================================
    # xes methods
    # =============================================================================

    def get_xes_events_data(self):
        TIMESTEP_KEY = 'time:timestamp'

        log = pm4py.read_xes(self.input)
        try:
            source = log.attributes['source']
        except KeyError:
            source = ''
        flattern_log = ([{**event, 'caseid': trace.attributes['concept:name']} for trace in log for event in trace])
        temp_data = pd.DataFrame(flattern_log)
        temp_data[TIMESTEP_KEY] = temp_data.apply(lambda x: x[TIMESTEP_KEY].strftime(self.timeformat), axis=1)
        temp_data[TIMESTEP_KEY] = pd.to_datetime(temp_data[TIMESTEP_KEY], format=self.timeformat)
        temp_data.rename(columns={'concept:name': 'task', 'lifecycle:transition': 'event_type', 'org:resource': 'user',
                                  ('%s' % TIMESTEP_KEY): 'timestamp'}, inplace=True)
        temp_data = temp_data[~temp_data.task.isin(['Start', 'End', 'start', 'end'])].reset_index(drop=True)
        temp_data = temp_data[temp_data.event_type.isin(['start', 'complete'])].reset_index(drop=True)
        if source == 'com.qbpsimulator' and len(temp_data.iloc[0].elementId.split('_')) > 1:
            temp_data['etype'] = temp_data.apply(lambda x: x.elementId.split('_')[0], axis=1)
            temp_data = (temp_data[temp_data.etype == 'Task'].reset_index(drop=True))
        self.raw_data = temp_data.to_dict('records')
        if self.verbose:
            sup.print_performed_task('Rearranging log traces ')
        self.data = self.reorder_xes(temp_data)
        self.data = pd.DataFrame(self.data)
        self.data.drop_duplicates(inplace=True)
        self.data = self.data.to_dict('records')
        self.append_csv_start_end()
        if self.verbose:
            sup.print_done_task()

    def reorder_xes(self, temp_data):
        """
        this method match the duplicated events on the .xes log
        """
        temp_data = pd.DataFrame(temp_data)
        ordered_event_log = list()
        if self.one_timestamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            temp_data = temp_data[temp_data.event_type == 'complete']
            ordered_event_log = temp_data.rename(
                columns={'timestamp': 'end_timestamp'})
            ordered_event_log = ordered_event_log.drop(columns='event_type')
            ordered_event_log = ordered_event_log.to_dict('records')
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            for caseid, group in temp_data.groupby(by=['caseid']):
                trace = group.to_dict('records')
                temp_trace = list()
                incomplete = False
                for i in range(0, len(trace) - 1):
                    incomplete = False
                    if trace[i]['event_type'] == 'start':
                        c_task_name = trace[i]['task']
                        remaining = trace[i + 1:]
                        complete_event = next((event for event in remaining if
                                               (event['task'] == c_task_name and event['event_type'] == 'complete')),
                                              None)
                        if complete_event:
                            temp_trace.append(
                                {'caseid': caseid,
                                 'task': trace[i]['task'],
                                 'user': trace[i]['user'],
                                 'start_timestamp': trace[i]['timestamp'],
                                 'end_timestamp': complete_event['timestamp']})
                        else:
                            incomplete = True
                            break
                if not incomplete:
                    ordered_event_log.extend(temp_trace)
        return ordered_event_log

    # =============================================================================
    # csv methods
    # =============================================================================
    def get_csv_events_data(self):
        """
        reads and parse all the events information from a csv file
        """
        if self.verbose:
            sup.print_performed_task('Reading log traces ')
        log = pd.read_csv(self.input)
        if self.one_timestamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user', 'end_timestamp']]
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user',
                           'start_timestamp', 'end_timestamp']]
            log['start_timestamp'] = pd.to_datetime(log['start_timestamp'],
                                                    format=self.timeformat)
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        self.data = log.to_dict('records')
        self.append_csv_start_end()
        self.split_event_transitions()
        if self.verbose:
            sup.print_done_task()

    def split_event_transitions(self):
        temp_raw = list()
        if self.one_timestamp:
            for event in self.data:
                temp_event = event.copy()
                temp_event['timestamp'] = temp_event.pop('end_timestamp')
                temp_event['event_type'] = 'complete'
                temp_raw.append(temp_event)
        else:
            for event in self.data:
                start_event = event.copy()
                complete_event = event.copy()
                start_event.pop('end_timestamp')
                complete_event.pop('start_timestamp')
                start_event['timestamp'] = start_event.pop('start_timestamp')
                complete_event['timestamp'] = complete_event.pop('end_timestamp')
                start_event['event_type'] = 'start'
                complete_event['event_type'] = 'complete'
                temp_raw.append(start_event)
                temp_raw.append(complete_event)
        self.raw_data = temp_raw

    def append_csv_start_end(self):
        end_start_times = dict()
        for case, group in pd.DataFrame(self.data).groupby('caseid'):
            end_start_times[(case, 'Start')] = (
                    group.start_timestamp.min() - timedelta(microseconds=1))
            end_start_times[(case, 'End')] = (
                    group.end_timestamp.max() + timedelta(microseconds=1))
        new_data = list()
        data = sorted(self.data, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = end_start_times[(key, new_event)]
                if not self.one_timestamp:
                    temp_event['start_timestamp'] = end_start_times[(key, new_event)]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self.data = new_data

    # =============================================================================
    # Accessor methods
    # =============================================================================
    def get_traces(self):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in self.data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    def get_raw_traces(self):
        """
        returns the raw data splitted by caseid and ordered by timestamp
        """
        cases = list(set([c['caseid'] for c in self.raw_data]))
        traces = list()
        for case in cases:
            trace = sorted(
                list(filter(lambda x, case=case: (x['caseid'] == case), self.raw_data)),
                key=itemgetter('timestamp'))
            traces.append(trace)
        return traces

    def set_data(self, data):
        """
        seting method for the data attribute
        """
        self.data = data

    # =============================================================================
    # Support Method
    # =============================================================================
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension in ['.xes', '.csv', '.mxml']:
            filename = filename + file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(outFileName)
        elif file_extension == '.zip':
            filename, file_extension = self.decompress_file_zip(filename)
        else:
            raise IOError('file type not supported')
        return filename, file_extension

    # Decompress .gz files
    def decompress_file_gzip(self, outFileName):
        inFile = gzip.open(self.input, 'rb')
        outFile = open(outFileName, 'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName, fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, outfilename):
        with zf.ZipFile(self.input, "r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension

    @staticmethod
    def copy_without_data(log: 'LogReader') -> 'LogReader':
        default_options = ReadOptions(column_names=ReadOptions.column_names_default())
        copy = LogReader(input=log.input, settings=default_options, load=False)
        copy.timeformat = log.timeformat
        copy.column_names = log.column_names
        copy.one_timestamp = log.one_timestamp
        copy.filter_d_attrib = log.filter_d_attrib
        copy.verbose = log.verbose
        return copy


class LogReader(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    def __init__(self, input: Union[Path, str], settings: ReadOptions, verbose=True, load=True):
        if isinstance(input, Path):
            self.input = input.absolute().__str__()
        else:
            self.input = input
        self.file_name, self.file_extension = self.define_ftype()

        self.timeformat = settings.timeformat
        self.column_names = settings.column_names
        self.one_timestamp = settings.one_timestamp  # TODO: remove everywhere, interval log is compulsory
        self.filter_d_attrib = settings.filter_d_attrib
        self.verbose = verbose
        self.data = list()
        self.raw_data = list()

        if load:
            self.load_data_from_file()

    @staticmethod
    def copy_without_data(log: 'LogReader') -> 'LogReader':
        default_options = ReadOptions(column_names=ReadOptions.column_names_default())
        copy = LogReader(input=log.input, settings=default_options, load=False)
        copy.timeformat = log.timeformat
        copy.column_names = log.column_names
        copy.one_timestamp = log.one_timestamp
        copy.filter_d_attrib = log.filter_d_attrib
        copy.verbose = log.verbose
        return copy

    def load_data_from_file(self):
        """
        reads all the data from the log depending
        the extension of the file
        """
        # TODO: esto se puede manejar mejor con un patron fabrica
        if self.file_extension == '.xes':
            self.get_xes_events_data()
        elif self.file_extension == '.csv':
            self.get_csv_events_data()

    def get_xes_events_data(self):
        timestamp_key = 'time:timestamp'

        log = pm4py.read_xes(self.input)
        try:
            source = log.attributes['source']
        except KeyError:
            source = ''

        log = interval_lifecycle.to_interval(log)

        flattened_log = [{**event, 'caseid': trace.attributes['concept:name']} for trace in log for event in trace]
        temp_data = pd.DataFrame(flattened_log)
        # NOTE: takes up more memory
        # temp_data = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        # NOTE: Stripping zone information from the log
        # TODO: is it applicable here: pm4py.objects.log.util.dataframe_utils.convert_timestamp_columns_in_df?
        temp_data[timestamp_key] = temp_data.apply(lambda x: x[timestamp_key].strftime(self.timeformat), axis=1)
        temp_data[timestamp_key] = pd.to_datetime(temp_data[timestamp_key], format=self.timeformat)
        temp_data['start_timestamp'] = temp_data.apply(lambda x: x['start_timestamp'].strftime(self.timeformat), axis=1)
        temp_data['start_timestamp'] = pd.to_datetime(temp_data['start_timestamp'], format=self.timeformat)

        # NOTE: Converting all timestamps to UTC
        # temp_data['start_timestamp'] = pd.to_datetime(temp_data['start_timestamp'], utc=True)
        # temp_data[timestamp_key] = pd.to_datetime(temp_data[timestamp_key], utc=True)

        temp_data.rename(columns={
            'concept:name': 'task',
            'case:concept:name': 'caseid',
            'lifecycle:transition': 'event_type',
            'org:resource': 'user',
            timestamp_key: 'end_timestamp'
        }, inplace=True)

        temp_data = temp_data[~temp_data.task.isin(['Start', 'End', 'start', 'end'])].reset_index(drop=True)

        if source == 'com.qbpsimulator' and len(temp_data.iloc[0].elementId.split('_')) > 1:
            temp_data['etype'] = temp_data.apply(lambda x: x.elementId.split('_')[0], axis=1)
            temp_data = (temp_data[temp_data.etype == 'Task'].reset_index(drop=True))

        self.raw_data = temp_data.to_dict('records')

        temp_data.drop_duplicates(inplace=True)

        self.data = temp_data.to_dict('records')

        self.append_csv_start_end()

        if self.verbose:
            sup.print_done_task()

    def get_csv_events_data(self):  # TODO: does this function work? Had problems reading in Production.csv
        """
        reads and parse all the events information from a csv file
        """
        if self.verbose:
            sup.print_performed_task('Reading log traces ')
        log = pd.read_csv(self.input)
        if self.one_timestamp:
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user', 'end_timestamp']]
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        else:
            self.column_names['Start Timestamp'] = 'start_timestamp'
            self.column_names['Complete Timestamp'] = 'end_timestamp'
            log = log.rename(columns=self.column_names)
            log = log.astype({'caseid': object})
            log = (log[(log.task != 'Start') & (log.task != 'End')]
                   .reset_index(drop=True))
            if self.filter_d_attrib:
                log = log[['caseid', 'task', 'user',
                           'start_timestamp', 'end_timestamp']]
            log['start_timestamp'] = pd.to_datetime(log['start_timestamp'],
                                                    format=self.timeformat)
            log['end_timestamp'] = pd.to_datetime(log['end_timestamp'],
                                                  format=self.timeformat)
        self.data = log.to_dict('records')
        self.append_csv_start_end()
        if self.verbose:
            sup.print_done_task()

    def append_csv_start_end(self):
        end_start_times = dict()
        for case, group in pd.DataFrame(self.data).groupby('caseid'):
            end_start_times[(case, 'Start')] = group.start_timestamp.min() - timedelta(microseconds=1)
            end_start_times[(case, 'End')] = group.end_timestamp.max() + timedelta(microseconds=1)
        new_data = list()
        data = sorted(self.data, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = end_start_times[(key, new_event)]
                if not self.one_timestamp:
                    temp_event['start_timestamp'] = end_start_times[(key, new_event)]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self.data = new_data

    def get_traces(self):  # TODO: can we do it just with Pandas functions?
        """
        Returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in self.data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), self.data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

    # NOTE: needed only for TracesAligner
    def get_raw_traces(self):
        """
        returns the raw data splitted by caseid and ordered by timestamp
        """
        cases = list(set([c['caseid'] for c in self.raw_data]))
        traces = list()
        for case in cases:
            trace = sorted(
                list(filter(lambda x, case=case: (x['caseid'] == case), self.raw_data)),
                key=itemgetter('end_timestamp'))
            traces.append(trace)
        return traces

    def set_data(self, data):
        self.data = data

    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension in ['.xes', '.csv', '.mxml']:
            filename = filename + file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(outFileName)
        elif file_extension == '.zip':
            filename, file_extension = self.decompress_file_zip(filename)
        else:
            raise IOError('file type not supported')
        return filename, file_extension

    # Decompress .gz files
    def decompress_file_gzip(self, outFileName):
        inFile = gzip.open(self.input, 'rb')
        outFile = open(outFileName, 'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName, fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, outfilename):
        with zf.ZipFile(self.input, "r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension

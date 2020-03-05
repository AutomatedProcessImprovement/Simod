# -*- coding: utf-8 -*-
import datetime
import xml.etree.ElementTree as ET
import gzip
import zipfile as zf
import os
import itertools as it

import pandas as pd
from operator import itemgetter

from support_modules import support as sup


class LogReader(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    def __init__(self, input, settings):
        """constructor"""
        self.input = input
        self.file_name, self.file_extension = self.define_ftype()

        self.timeformat = settings['timeformat']
        self.column_names = settings['column_names']
        self.one_timestamp = settings['one_timestamp']
        self.filter_d_attrib = settings['filter_d_attrib']
        self.ns_include = settings['ns_include']

        self.data = list()
        self.raw_data = list()
        self.load_data_from_file()

    def load_data_from_file(self):
        """
        reads all the data from the log depending
        the extension of the file
        """
        # TODO: esto se puede manejar mejor con un patron de diseno
        if self.file_extension == '.xes':
            self.get_xes_events_data()
        elif self.file_extension == '.csv':
            self.get_csv_events_data()
        # elif self.file_extension == '.mxml':
        #     self.data, self.raw_data = self.get_mxml_events_data()

# =============================================================================
# xes methods
# =============================================================================
    def get_xes_events_data(self):
        """
        reads and parse all the events information from a xes file
        """
        temp_data = list()
        tree = ET.parse(self.input)
        root = tree.getroot()
        if self.ns_include:
            ns = {'xes': root.tag.split('}')[0].strip('{')}
            tags = dict(trace='xes:trace',
                        string='xes:string',
                        event='xes:event',
                        date='xes:date')
        else:
            ns = {'xes': ''}
            tags = dict(trace='trace',
                        string='string',
                        event='event',
                        date='date')
        traces = root.findall(tags['trace'], ns)
        i = 0
        sup.print_performed_task('Reading log traces ')
        for trace in traces:
            temp_trace = list()
            caseid = ''
            for string in trace.findall(tags['string'], ns):
                if string.attrib['key'] == 'concept:name':
                    caseid = string.attrib['value']
            for event in trace.findall(tags['event'], ns):
                task = ''
                user = ''
                event_type = ''
                for string in event.findall(tags['string'], ns):
                    if string.attrib['key'] == 'concept:name':
                        task = string.attrib['value']
                    if string.attrib['key'] == 'org:resource':
                        user = string.attrib['value']
                    if string.attrib['key'] == 'lifecycle:transition':
                        event_type = string.attrib['value'].lower()
                timestamp = ''
                for date in event.findall(tags['date'], ns):
                    if date.attrib['key'] == 'time:timestamp':
                        timestamp = date.attrib['value']
                        try:
                            timestamp = datetime.datetime.strptime(
                                timestamp[:-6], self.timeformat)
                        except ValueError:
                            timestamp = datetime.datetime.strptime(
                                timestamp, self.timeformat)
                # By default remove Start and End events
                # but will be added to standardize
                if task not in ['0', '-1', 'Start', 'End', 'start', 'end']:
                    if ((not self.one_timestamp) or
                        (self.one_timestamp and event_type == 'complete')):
                        temp_trace.append(dict(caseid=caseid,
                                               task=task,
                                               event_type=event_type,
                                               user=user,
                                               timestamp=timestamp))
            if temp_trace:
                temp_trace = self.append_xes_start_end(temp_trace)
            temp_data.extend(temp_trace)
            i += 1
        self.raw_data = temp_data
        self.data = self.reorder_xes(temp_data)
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
            cases = temp_data.caseid.unique()
            for case in cases:
                start_ev = (temp_data[(temp_data.event_type == 'start') &
                                      (temp_data.caseid == case)]
                            .sort_values(by='timestamp', ascending=True)
                            .to_dict('records'))
                complete_ev = (temp_data[(temp_data.event_type == 'complete') &
                                         (temp_data.caseid == case)]
                               .sort_values(by='timestamp', ascending=True)
                               .to_dict('records'))
                if len(start_ev) == len(complete_ev):
                    temp_trace = list()
                    for i, _ in enumerate(start_ev):
                        match = False
                        for j, _ in enumerate(complete_ev):
                            if start_ev[i]['task'] == complete_ev[j]['task']:
                                temp_trace.append(
                                    {'caseid': case,
                                     'task': start_ev[i]['task'],
                                     'user': start_ev[i]['user'],
                                     'start_timestamp': start_ev[i]['timestamp'],
                                     'end_timestamp': complete_ev[j]['timestamp']})
                                match = True
                                break
                        if match:
                            del complete_ev[j]
                    if match:
                        ordered_event_log.extend(temp_trace)
        return ordered_event_log

    def append_xes_start_end(self, trace):
        for event in ['Start', 'End']:
            idx = 0 if event == 'Start' else -1
            complete_ev = dict()
            complete_ev['caseid'] = trace[idx]['caseid']
            complete_ev['task'] = event
            complete_ev['event_type'] = 'complete'
            complete_ev['user'] = event
            complete_ev['timestamp'] = trace[idx]['timestamp']
            if event == 'Start':
                trace.insert(0, complete_ev)
                if not self.one_timestamp:
                    start_ev = complete_ev.copy()
                    start_ev['event_type'] = 'start'
                    trace.insert(0, start_ev)
            else:
                trace.append(complete_ev)
                if not self.one_timestamp:
                    start_ev = complete_ev.copy()
                    start_ev['event_type'] = 'start'
                    trace.insert(-1, start_ev)
        return trace

# =============================================================================
# csv methods
# =============================================================================
    def get_csv_events_data(self):
        """
        reads and parse all the events information from a csv file
        """
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
        new_data = list()
        data = sorted(self.data, key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                t_key = 'end_timestamp'
                if not self.one_timestamp and new_event == 'Start':
                    t_key = 'start_timestamp'
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = trace[idx][t_key]
                if not self.one_timestamp:
                    temp_event['start_timestamp'] = trace[idx][t_key]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self.data = new_data

# =============================================================================
# Accesssor methods
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
                list(filter(lambda x: (x['caseid'] == case), self.raw_data)),
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
            file_extension = file_extension
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

#     def get_mxml_events_data(self, filename, parameters):
#         """read and parse all the events information from a MXML file"""
#         temp_data = list()
#         tree = ET.parse(filename)
#         root = tree.getroot()
#         process = root.find('Process')
#         procInstas = process.findall('ProcessInstance')
#         i = 0
#         for procIns in procInstas:
#             sup.print_progress(((i / (len(procInstas) - 1)) * 100), 'Reading log traces ')
#             caseid = procIns.get('id')
#             auditTrail = procIns.findall('AuditTrailEntry')
#             for trail in auditTrail:
#                 task = ''
#                 user = ''
#                 event_type = ''
#                 timestamp = ''
#                 attributes = trail.find('Data').findall('Attribute')
#                 for attr in attributes:
#                     if (attr.get('name') == 'concept:name'):
#                         task = attr.text
#                     if (attr.get('name') == 'lifecycle:transition'):
#                         event_type = attr.text
#                     if (attr.get('name') == 'org:resource'):
#                         user = attr.text
#                 event_type = trail.find('EventType').text
#                 timestamp = trail.find('Timestamp').text
#                 timestamp = datetime.datetime.strptime(trail.find('Timestamp').text[:-6], parameters['timeformat'])
#                 temp_data.append(
#                     dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
#                          end_timestamp=timestamp))
#             i += 1
#         raw_data = temp_data
#         temp_data = self.reorder_mxml(temp_data)
#         sup.print_done_task()
#         return temp_data, raw_data
#     def reorder_mxml(self, temp_data):
#         """this method joints the duplicated events on the .mxml log"""
#         data = list()
#         start_events = list(filter(lambda x: x['event_type'] == 'start', temp_data))
#         finish_events = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
#         for x, y in zip(start_events, finish_events):
#             data.append(dict(caseid=x['caseid'], task=x['task'], event_type=x['event_type'],
#                              user=x['user'], start_timestamp=x['start_timestamp'], end_timestamp=y['start_timestamp']))
#         return data
#     # TODO manejo de excepciones
#     def find_first_task(self):
#         """finds the first task"""
#         cases = list()
#         [cases.append(c['caseid']) for c in self.data]
#         cases = sorted(list(set(cases)))
#         first_task_names = list()
#         for case in cases:
#             trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter('start_timestamp'))
#             first_task_names.append(trace[0]['task'])
#         first_task_names = list(set(first_task_names))
#         return first_task_names
#     def read_resource_task(self,task,roles):
#         """returns the resource that performs a task"""
#         filtered_list = list(filter(lambda x: x['task']==task, self.data))
#         role_assignment = list()
#         for task in filtered_list:
#             for role in roles:
#                 for member in role['members']:
#                     if task['user']==member:
#                         role_assignment.append(role['role'])
#         return max(role_assignment)

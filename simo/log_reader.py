# -*- coding: utf-8 -*-
import csv
import datetime
from datetime import date
import xml.etree.ElementTree as ET
import gzip
import zipfile as zf
import os
from operator import itemgetter

import support as sup


class LogReader(object):
    """
	This class reads and parse the elements of a given process log in format .xes or .csv
	"""

    def __init__(self, input,log_columns_numbers, start_timeformat, end_timeformat):
        """constructor"""
        self.input = input
        self.data, self.raw_data = self.load_data_from_file(log_columns_numbers, start_timeformat, end_timeformat)


    # Support Method
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(self.input, outFileName)
        if file_extension=='.zip':
            filename,file_extension = self.decompress_file_zip(self.input, filename)
        if not (file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml'):
            raise IOError('file type not supported')
        return filename,file_extension

    # Decompress .gz files
    def decompress_file_gzip(self,filename, outFileName):
        inFile = gzip.open(filename, 'rb')
        outFile = open(outFileName,'wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        _, fileExtension = os.path.splitext(outFileName)
        return outFileName,fileExtension

    # Decompress .zip files
    def decompress_file_zip(self, filename, outfilename):
        with zf.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall("../inputs/")
        _, fileExtension = os.path.splitext(outfilename)
        return outfilename, fileExtension

    # Reading methods
    def load_data_from_file(self, log_columns_numbers, start_timeformat, end_timeformat):
        """reads all the data from the log depending the extension of the file"""
        temp_data = list()
        filename, file_extension = self.define_ftype()
        if file_extension == '.xes':
            temp_data, raw_data = self.get_xes_events_data(filename,start_timeformat, end_timeformat)
        elif file_extension == '.csv':
            temp_data = self.get_csv_events_data(log_columns_numbers, start_timeformat, end_timeformat)
        elif file_extension == '.mxml':
            temp_data, raw_data = self.get_mxml_events_data(filename,start_timeformat, end_timeformat)
        return temp_data, raw_data

    def get_xes_events_data(self, filename,start_timeformat, end_timeformat):
        """reads and parse all the events information from a xes file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        ns = {'xes': 'http://www.xes-standard.org'}
        traces = root.findall('xes:trace', ns)
        i = 0
        for trace in traces:
            sup.print_progress(((i / (len(traces) - 1)) * 100), 'Reading log traces ')
            caseid = ''
            for string in trace.findall('xes:string', ns):
                if string.attrib['key'] == 'concept:name':
                    caseid = string.attrib['value']
            for event in trace.findall('xes:event', ns):
                task = ''
                user = ''
                event_type = ''
                complete_timestamp = ''
                for string in event.findall('xes:string', ns):
                    if string.attrib['key'] == 'concept:name':
                        task = string.attrib['value']
                    if string.attrib['key'] == 'org:resource':
                        user = string.attrib['value']
                    if string.attrib['key'] == 'lifecycle:transition':
                        event_type = string.attrib['value']
                    if string.attrib['key'] == 'Complete_Timestamp':
                        complete_timestamp = string.attrib['value']
                        if complete_timestamp != 'End':
                            complete_timestamp = datetime.datetime.strptime(complete_timestamp, end_timeformat)
                timestamp = ''
                for date in event.findall('xes:date', ns):
                    if date.attrib['key'] == 'time:timestamp':
                        timestamp = date.attrib['value']
                        timestamp = datetime.datetime.strptime(timestamp[:-6], start_timeformat)
                temp_data.append(
                    dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
                         end_timestamp=complete_timestamp))
            i += 1
        raw_data = temp_data
        temp_data = self.reorder_xes(temp_data)
        sup.print_done_task()
        return temp_data, raw_data

    def reorder_xes(self, temp_data):
        """this method joints the duplicated events on the .xes log"""
        data = list()
        start_events = list(filter(lambda x: x['event_type'] == 'start', temp_data))
        finish_events = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
        for x, y in zip(start_events, finish_events):
            data.append(dict(caseid=x['caseid'], task=x['task'], event_type=x['task'],
                             user=x['user'], start_timestamp=x['start_timestamp'], end_timestamp=y['start_timestamp']))
        return data

    def get_mxml_events_data(self, filename,start_timeformat, end_timeformat):
        """read and parse all the events information from a MXML file"""
        temp_data = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        process = root.find('Process')
        procInstas = process.findall('ProcessInstance')
        i = 0
        for procIns in procInstas:
            sup.print_progress(((i / (len(procInstas) - 1)) * 100), 'Reading log traces ')
            caseid = procIns.get('id')
            complete_timestamp = ''
            auditTrail = procIns.findall('AuditTrailEntry')
            for trail in auditTrail:
                task = ''
                # user = ''
                event_type = ''
                # type_task = ''
                timestamp = ''
                originator = ''
                task = trail.find('WorkflowModelElement').text
                event_type = trail.find('EventType').text
                timestamp = trail.find('Timestamp').text
                originator = trail.find('Originator').text
                timestamp = datetime.datetime.strptime(trail.find('Timestamp').text[:-6], start_timeformat)
                temp_data.append(
                    dict(caseid=caseid, task=task, event_type=event_type, user=originator, start_timestamp=timestamp,
                         end_timestamp=timestamp))

            i += 1
        raw_data = temp_data
        temp_data = self.reorder_mxml(temp_data)
        sup.print_done_task()
        return temp_data, raw_data

    def reorder_mxml(self, temp_data):
        """this method joints the duplicated events on the .mxml log"""
        data = list()
        start_events = list(filter(lambda x: x['event_type'] == 'start', temp_data))
        finish_events = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
        for x, y in zip(start_events, finish_events):
            data.append(dict(caseid=x['caseid'], task=x['task'], event_type=x['event_type'],
                             user=x['user'], start_timestamp=x['start_timestamp'], end_timestamp=y['start_timestamp']))
        return data

    def get_csv_events_data(self, log_columns_numbers, start_timeformat, end_timeformat):
        """reads and parse all the events information from a csv file"""
        flength = file_size(self.input)
        i = 0
        temp_data = list()
        with open(self.input, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(filereader, None)  # skip the headers
            for row in filereader:
                sup.print_progress(((i / (flength - 1)) * 100), 'Reading log traces ')
                timestamp = ''
                complete_timestamp = ''
                if row[log_columns_numbers[1]] != 'End':
                    timestamp = datetime.datetime.strptime(row[log_columns_numbers[4]], start_timeformat)
                    complete_timestamp = datetime.datetime.strptime(row[log_columns_numbers[5]], end_timeformat)
                temp_data.append(dict(caseid=row[log_columns_numbers[0]], task=row[log_columns_numbers[1]],
                                      event_type=row[log_columns_numbers[2]], user=row[log_columns_numbers[3]],
                                      start_timestamp=timestamp, end_timestamp=complete_timestamp))
                i += 1
        return temp_data

    # TODO manejo de excepciones
    def find_first_task(self):
        """finds the first task"""
        cases = list()
        [cases.append(c['caseid']) for c in self.data]
        cases = sorted(list(set(cases)))
        first_task_names = list()
        for case in cases:
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter('start_timestamp'))
            first_task_names.append(trace[0]['task'])
        first_task_names = list(set(first_task_names))
        return first_task_names

    def get_traces(self):
        """returns the data splitted by caseid and ordered by start_timestamp"""
        cases = list()
        for c in self.data: cases.append(c['caseid'])
        cases = sorted(list(set(cases)))
        traces = list()
        for case in cases:
            # trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.data)), key=itemgetter('start_timestamp'))
            trace = list(filter(lambda x: (x['caseid'] == case), self.data))
            traces.append(trace)
        return traces

    def get_raw_traces(self):
        """returns the raw data splitted by caseid and ordered by start_timestamp"""
        cases = list()
        for c in self.raw_data: cases.append(c['caseid'])
        cases = sorted(list(set(cases)))
        traces = list()
        for case in cases:
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), self.raw_data)), key=itemgetter('start_timestamp'))
            traces.append(trace)
        return traces

    def read_resource_task(self,task,roles):
        """returns the resource that performs a task"""
        filtered_list = list(filter(lambda x: x['task']==task, self.data))
        role_assignment = list()
        for task in filtered_list:
            for role in roles:
                for member in role['members']:
                    if task['user']==member:
                        role_assignment.append(role['role'])
        return max(role_assignment)

    def set_data(self,data):
        """seting method for the data attribute"""
        self.data = data

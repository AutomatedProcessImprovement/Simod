# -*- coding: utf-8 -*-
import csv
import datetime
#import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
import gzip
import zipfile as zf
import os
from operator import itemgetter


from support_modules import support as sup


class LogReader(object):
    """
	This class reads and parse the elements of a given process log in format .xes or .csv
	"""

    def __init__(self, input, timeformat, ns_include=True, one_timestamp=False):
        """constructor"""
        self.input = input
        self.data, self.raw_data = self.load_data_from_file(timeformat, ns_include, one_timestamp)


    # Support Method
    def define_ftype(self):
        filename, file_extension = os.path.splitext(self.input)
        if file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml' :
             filename = filename + file_extension
             file_extension = file_extension
        elif file_extension == '.gz':
            outFileName = filename
            filename, file_extension = self.decompress_file_gzip(self.input, outFileName)
        elif file_extension=='.zip':
            filename,file_extension = self.decompress_file_zip(self.input, filename)
        elif not (file_extension == '.xes' or file_extension == '.csv' or file_extension == '.mxml'):
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
    def load_data_from_file(self, timeformat, ns_include, one_timestamp):
        """reads all the data from the log depending the extension of the file"""
        temp_data = list()
        filename, file_extension = self.define_ftype()
        if file_extension == '.xes':
            temp_data, raw_data = self.get_xes_events_data(filename,timeformat, ns_include, one_timestamp)
        elif file_extension == '.csv':
            temp_data, raw_data = self.get_csv_events_data(timeformat)
        elif file_extension == '.mxml':
            temp_data, raw_data = self.get_mxml_events_data(filename,timeformat)
        return temp_data, raw_data

    def get_xes_events_data(self, filename,timeformat, ns_include, one_timestamp):
        """reads and parse all the events information from a xes file"""
        tree = ET.parse(filename)
        root = tree.getroot()
        if ns_include:
            ns = {'xes': root.tag.split('}')[0].strip('{')}
            tags = dict(trace='xes:trace',string='xes:string',event='xes:event',date='xes:date')
        else:
            ns = {'xes':''}
            tags = dict(trace='trace',string='string',event='event',date='date')
        traces = root.findall(tags['trace'], ns)
        sup.print_performed_task('Reading log traces ')
        for trace in traces:
            caseid = ''
            for string in trace.findall(tags['string'], ns):
                if string.attrib['key'] == 'concept:name':
                    caseid = string.attrib['value']
                    break
            print(caseid)
#            for event in trace.findall(tags['event'], ns):
#                task = ''
#                user = ''
#                event_type = ''
#                complete_timestamp = ''
#                for string in event.findall(tags['string'], ns):
#                    if string.attrib['key'] == 'concept:name':
#                        task = string.attrib['value']                        
#                    if string.attrib['key'] == 'org:resource':
#                        user = string.attrib['value']
#                    if string.attrib['key'] == 'lifecycle:transition':
#                        event_type = string.attrib['value'].lower()
#                    if string.attrib['key'] == 'Complete_Timestamp':
#                        complete_timestamp = string.attrib['value']
#                        if complete_timestamp != 'End':
#                            complete_timestamp = datetime.datetime.strptime(complete_timestamp, timeformat)
#                timestamp = ''
#                for date in event.findall(tags['date'], ns):
#                    if date.attrib['key'] == 'time:timestamp':
#                        timestamp = date.attrib['value']
#                        try:
#                            timestamp = datetime.datetime.strptime(timestamp[:-6], timeformat)
#                        except ValueError:
#                            timestamp = datetime.datetime.strptime(timestamp, timeformat)
#                if not (task == '0' or task == '-1'):
#                    temp_data.append(
#                        dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
#                             end_timestamp=complete_timestamp))
#            i += 1
#        raw_data = temp_data
#        temp_data = self.reorder_xes(temp_data, one_timestamp)
#        sup.print_done_task()
#        return temp_data, raw_data

    def reorder_xes(self, temp_data, one_timestamp):
        """this method joints the duplicated events on the .xes log"""
        ordered_event_log = list()
        if one_timestamp:
            ordered_event_log = list(filter(lambda x: x['event_type'] == 'complete', temp_data))
            for event in ordered_event_log:
                event['end_timestamp'] = event['start_timestamp']
        else:
            events = list(filter(lambda x: (x['event_type'] == 'start' or x['event_type'] == 'complete'), temp_data))
            cases = list({x['caseid'] for x in events})
            for case in cases:
                start_events = sorted(list(filter(lambda x: x['event_type'] == 'start' and x['caseid'] == case, events)), key=lambda x:x['start_timestamp'])
                finish_events = sorted(list(filter(lambda x: x['event_type'] == 'complete' and x['caseid'] == case, events)), key=lambda x:x['start_timestamp'])
                if len(start_events) == len(finish_events):
                    temp_trace = list()
                    for i, _ in enumerate(start_events):
                        match = False
                        for j, _ in enumerate(finish_events):
                            if start_events[i]['task'] == finish_events[j]['task']:
                                temp_trace.append(dict(caseid=case, task=start_events[i]['task'], event_type=start_events[i]['task'],
                                     user=start_events[i]['user'], start_timestamp=start_events[i]['start_timestamp'], end_timestamp=finish_events[j]['start_timestamp']))
                                match = True
                                break
                        if match:
                            del finish_events[j]
                    if match:
                        ordered_event_log.extend(temp_trace)
        return ordered_event_log

    def get_mxml_events_data(self, filename,timeformat):
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
            auditTrail = procIns.findall('AuditTrailEntry')
            for trail in auditTrail:
                task = ''
                user = ''
                event_type = ''
                timestamp = ''
                attributes = trail.find('Data').findall('Attribute')
                for attr in attributes:
                    if (attr.get('name') == 'concept:name'):
                        task = attr.text
                    if (attr.get('name') == 'lifecycle:transition'):
                        event_type = attr.text
                    if (attr.get('name') == 'org:resource'):
                        user = attr.text
                event_type = trail.find('EventType').text
                timestamp = trail.find('Timestamp').text
                timestamp = datetime.datetime.strptime(trail.find('Timestamp').text[:-6], timeformat)
                temp_data.append(
                    dict(caseid=caseid, task=task, event_type=event_type, user=user, start_timestamp=timestamp,
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

#    def get_csv_events_data(self, log_columns_numbers, timeformat):
#        """reads and parse all the events information from a csv file"""
#        flength = sup.file_size(self.input)
#        i = 0
#        temp_data = list()
#        with open(self.input, 'r') as csvfile:
#            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
#            next(filereader, None)  # skip the headers
#            for row in filereader:
#                sup.print_progress(((i / (flength - 1)) * 100), 'Reading log traces ')
#                timestamp = ''
#                complete_timestamp = ''
#                if row[log_columns_numbers[1]] != 'End':
#                    timestamp = datetime.datetime.strptime(row[log_columns_numbers[4]], timeformat)
#                    complete_timestamp = datetime.datetime.strptime(row[log_columns_numbers[5]], timeformat)
#                temp_data.append(dict(caseid=row[log_columns_numbers[0]], task=row[log_columns_numbers[1]],
#                                      event_type=row[log_columns_numbers[2]], user=row[log_columns_numbers[3]],
#                                      start_timestamp=timestamp, end_timestamp=complete_timestamp))
#                i += 1
#        return temp_data, temp_data

    def get_csv_events_data(self, timeformat):
        """reads and parse all the events information from a csv file"""
        flength = sup.file_size(self.input)
        i = 0
        temp_data = list()
        with open(self.input, 'r') as csvfile:
            filereader = csv.DictReader(csvfile, delimiter=',')
            for row in filereader:
                sup.print_progress(((i / (flength - 1)) * 100), 'Reading log traces ')
                timestamp = ''
                complete_timestamp = ''
                if row['task'] != 'End':
                    timestamp = datetime.datetime.strptime(row['start_timestamp'], timeformat)
                    complete_timestamp = datetime.datetime.strptime(row['end_timestamp'], timeformat)
                temp_data.append(dict(caseid=row['caseid'], task=row['task'],
                                      event_type=row['task'], user=row['resource'],
                                      start_timestamp=timestamp, end_timestamp=complete_timestamp))
                i += 1
            csvfile.close()
        return temp_data, temp_data


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

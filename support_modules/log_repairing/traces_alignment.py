# -*- coding: utf-8 -*-
import re
import datetime
import utils.support as sup
from operator import itemgetter
import subprocess
import os
import platform as pl


class TracesAligner(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    def __init__(self, log, not_conformant, settings):
        """constructor"""
        self.one_timestamp = settings['read_options']['one_timestamp']

        self.evaluate_alignment(settings)
        self.optimal_alignments = self.read_alignment_info(settings['aligninfo'])
        self.traces_alignments = self.traces_alignment_type(settings['aligntype'])

        self.traces = list()
        self.get_traces(log, not_conformant)

        self.aligned_traces = self.align_traces()

    def get_traces(self, log, not_conformant):
        nc_caseid = list(set([x[0]['caseid'] for x in not_conformant]))
        if self.one_timestamp:
            traces = log.get_traces()
        else:
            traces = log.get_raw_traces()
        self.traces = list(
            filter(lambda x: x[0]['caseid'] in nc_caseid, traces))

    def align_traces(self):
        """
        This method is the kernel of the alignment process
        """
        aligned_traces = list()
        i = 0
        size = len(self.traces)
        for trace in self.traces:
            # Remove Start and End events
            trace = [x for x in trace if x['task'] not in ['Start', 'End']]
            try:
                # Alignment of each trace
                aligned_trace = self.process_trace(trace)
                if self.one_timestamp:
                    aligned_trace = sorted(aligned_trace,
                                           key=itemgetter('end_timestamp'))
                    aligned_trace = self.append_start_end(aligned_trace)
                    aligned_traces.extend(aligned_trace)
                else:
                    # completeness check and reformating
                    aligned_trace = self.trace_verification(aligned_trace)
                    if aligned_trace:
                        aligned_trace = self.append_start_end(aligned_trace)
                        aligned_traces.extend(aligned_trace)
            except Exception as e:
                next
            sup.print_progress(((i / (size-1)) * 100),
                               'Aligning log traces with model ')
            i += 1
        sup.print_done_task()
        return aligned_traces

    def process_trace(self, trace):
        """
        This method performs the alignment of each trace,
        according with the data optimal alignment
        """
        # TODO: A matching algorithm maybe can be used to perform this operation
        caseid = trace[0]['caseid']
        alignment_data = list(filter(lambda x: x['caseid'] == str(caseid),
                                     self.traces_alignments))[0]
        aligned_trace = list()
        # If fitness is 1 all the trace is aligned
        if 0 < alignment_data['fitness'] < 1:
            # TODO: This reading can be more clear and made in just one step
            optimal_alignment = list(
                filter(lambda x: alignment_data['trace_type'] == x['trace_type'],
                       self.optimal_alignments))[0]['optimal_alignment']
            optimal_alignment = [x for x in optimal_alignment
                                 if x['task_name'] not in ['Start', 'End']]
            j = 0
            for i in range(0, len(optimal_alignment)):
                movement_type = optimal_alignment[i]['movement_type']
                # If the Model and the log are aligned copy the raw value
                if movement_type == 'LMGOOD':
                    aligned_trace.append(trace[j])
                    j += 1
                # If the Log needs an extra task, create the start and complete
                # events with time 0 and user AUTO
                elif movement_type == 'MREAL':
                    if i == 0 or not aligned_trace:
                        time = (trace[0]['end_timestamp']
                                if self.one_timestamp else trace[0]['timestamp'])
                    else:
                        time = (aligned_trace[-1]['end_timestamp']
                                if self.one_timestamp else aligned_trace[-1]['timestamp'])
                        time += datetime.timedelta(microseconds=1)
                    new_event = {'caseid':caseid,
                                 'task':optimal_alignment[i]['task_name'],
                                 'user':'AUTO'}
                    if self.one_timestamp:
                        new_event['end_timestamp'] = time
                    else:
                        new_event['timestamp'] = time
                        new_event['event_type'] = 'complete'
                    aligned_trace.append(new_event)
                    if not self.one_timestamp:
                        aligned_trace.append(
                            {'caseid': caseid,
                             'task': optimal_alignment[i]['task_name'],
                             'event_type': 'start',
                             'user': 'AUTO',
                             'timestamp': time})
                # If the event appears in the Log but not in the model
                # delete the event from the trace
                elif movement_type == 'L':
                    j += 1
        elif alignment_data['fitness'] == 1:
            aligned_trace = trace
        return aligned_trace

    def trace_verification(self, aligned_trace):
        """
        This method performs the completeness check,
        error correction and matches the start and complete events
        """
        tasks = list({x['task'] for x in aligned_trace})
        start_list = list(filter(lambda x: x['event_type'] == 'start',
                                 aligned_trace))
        complete_list = list(filter(lambda x: x['event_type'] == 'complete',
                                    aligned_trace))
        new_trace = list()
        missalignment = False
        for task in tasks:
            missalignment = False
            start_events = sorted(list(
                filter(lambda x: x['task'] == task, start_list)),
                key=itemgetter('timestamp'))
            complete_events = sorted(list(
                filter(lambda x: x['task'] == task, complete_list)),
                key=itemgetter('timestamp'))
            if(len(start_events) == len(complete_events)):
                for i, _ in enumerate(start_events):
                    new_trace.append({
                        'caseid': start_events[i]['caseid'],
                        'task': start_events[i]['task'],
                        'user': start_events[i]['user'],
                        'start_timestamp': start_events[i]['timestamp'],
                        'end_timestamp': complete_events[i]['timestamp']})
            else:
                missalignment = True
                break
        if not missalignment:
            new_trace = sorted(new_trace, key=itemgetter('start_timestamp'))
        else:
            new_trace = list()
        return new_trace

    def append_start_end(self, trace):
        """
        Addition of Start and End events to a trace

        Parameters
        ----------
        trace : list
        Returns
        -------
        trace : modified list with attached events

        """
        for new_event in ['Start', 'End']:
            idx = 0 if new_event == 'Start' else -1
            t_key = 'end_timestamp'
            if not self.one_timestamp and new_event == 'Start':
                t_key = 'start_timestamp'
            temp_event = dict()
            temp_event['caseid'] = trace[idx]['caseid']
            temp_event['task'] = new_event
            temp_event['user'] = new_event
            time = trace[idx][t_key] + datetime.timedelta(microseconds=1)
            if new_event == 'Start':
                time = trace[idx][t_key] - datetime.timedelta(microseconds=1)
            temp_event['end_timestamp'] = time
            if not self.one_timestamp:
                temp_event['start_timestamp'] = time
            if new_event == 'Start':
                trace.insert(0, temp_event)
            else:
                trace.append(temp_event)
        return trace

# =============================================================================
# External tool calling
# =============================================================================
# TODO: modify this three methods to create just one that evaluates and merge
# all the alignment data in just one structure

    def evaluate_alignment(self, settings):
        """
        Evaluate the traces alignment in relation with BPMN structure.

        Parameters
        ----------
        settings : Path to jar and file names

        """
        print(" -- Evaluating event log alignment --")
        file_name = settings['file'].split('.')[0]
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.append('-Xmx2G')
            args.append('-Xss8G')
        args.extend(['-jar', settings['align_path'],
                settings['output']+os.sep,
                file_name+'.xes',
                settings['file'].split('.')[0]+'.bpmn',
                'true'])
        subprocess.call(args, bufsize=-1)

# =============================================================================
# Support
# =============================================================================
    def read_alignment_info(self, filename):
        """
        Method for the reading of the alignment specification
        generated by the proconformance plug-in

        Parameters
        ----------
        filename : String

        Returns
        -------
        records : alignment info records
        """
        records = list()
        with open(filename) as fp:
            [next(fp) for i in range(3)]
            for line in fp:
                temp_record = line.split(',')
                trace_type = int(temp_record[0])
                optimal_alignment = list()
                for task in temp_record[2:]:
                    prog = re.compile('(\w+)(\()(.*)(\))')
                    result = prog.match(task)
                    if result.group(1) != 'MINVI':
                        optimal_alignment.append(
                            dict(movement_type=result.group(1),
                                 task_name=result.group(3).strip()))
                records.append(
                    dict(trace_type=trace_type,
                         optimal_alignment=optimal_alignment))
        return records

    def traces_alignment_type(self, filename):
        """
        Method for the reading of the alignment type
        generated by the proconformance plug-in

        Parameters
        ----------
        filename : String

        Returns
        -------
        records : alignment type
        """
        records = list()
        with open(filename) as fp:
            [next(fp) for i in range(7)]
            for line in fp:
                temp_record = line.split(',')
                records.append(dict(caseid=temp_record[2],
                                    trace_type=int(temp_record[1]),
                                    fitness=float(temp_record[11])))
        return records

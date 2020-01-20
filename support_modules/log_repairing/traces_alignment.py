# -*- coding: utf-8 -*-
import re
import datetime
from support_modules import support as sup
from operator import itemgetter
import subprocess
import os


def align_traces(log, settings, not_conformant):
    """this method is the kernel of the alignment process"""
    evaluate_alignment(settings)
    optimal_alignments = read_alignment_info(settings['aligninfo'])
    traces_alignments = traces_alignment_type(settings['aligntype'])

    nc_caseid = list(set([x[0]['caseid'] for x in not_conformant]))

    if settings['read_options']['one_timestamp']:
        traces = log.get_traces()
    else:
        traces = log.get_raw_traces()

    # nc_caseid = [92102]
    traces = list(filter(lambda x: x[0]['caseid'] in nc_caseid, traces))
    aligned_traces = list()
    i = 0
    size = len(traces)
    for trace in traces:
        # Remove Start and End events
        trace = [x for x in trace if x['task'] not in ['Start', 'End']]
        try:
            # Alignment of each trace
    
            aligned_trace = process_trace(trace, optimal_alignments,
                                          traces_alignments,
                                          settings['read_options']['one_timestamp'])
            if settings['read_options']['one_timestamp']:
                aligned_trace = sorted(aligned_trace, key=itemgetter('end_timestamp'))
                aligned_trace = append_start_end(aligned_trace, settings['read_options']['one_timestamp'])
                aligned_traces.extend(aligned_trace)
            else:
                # completeness check and reformating
                aligned_trace = trace_verification(aligned_trace, trace)
                if aligned_trace:
                    aligned_trace = append_start_end(aligned_trace, settings['read_options']['one_timestamp'])
                    aligned_traces.extend(aligned_trace)
        except Exception as e:
            next
        sup.print_progress(((i / (size-1))* 100), 'Aligning log traces with model ')
        i += 1
    sup.print_done_task()
    return aligned_traces

def process_trace(raw_trace, optimal_alignments, traces_alignments, one_timestamp):
    """this method performs the alignment of each trace according with the data optimal alignment"""
    caseid = raw_trace[0]['caseid']
    alignment_data = list(filter(lambda x: x['caseid'] == str(caseid), traces_alignments))[0]
    aligned_trace = list()
    # If fitness is 1 all the trace es aligned
    if 0 < alignment_data['fitness'] < 1:
        optimal_alignment = list(filter(lambda x: alignment_data['trace_type'] == x['trace_type'], optimal_alignments))[0]['optimal_alignment']
        optimal_alignment = [x for x in optimal_alignment if x['task_name'] not in ['Start', 'End']]
        j = 0
        for i in range(0,len(optimal_alignment)):
            movement_type = optimal_alignment[i]['movement_type']
            # If the Model and the log are aligned copy the raw value
            if movement_type =='LMGOOD':
                aligned_trace.append(raw_trace[j])
                j += 1
            # If the Log needs an extra task, create the start and complet event with time 0 and user AUTO
            elif movement_type =='MREAL':
                if i == 0 or not aligned_trace:
                    time = raw_trace[0]['end_timestamp'] if one_timestamp else raw_trace[0]['timestamp']
                else:
                    time = aligned_trace[-1]['end_timestamp'] if one_timestamp else aligned_trace[-1]['timestamp']
                    time += datetime.timedelta(microseconds=1)
                new_event = {'caseid':caseid, 'task':optimal_alignment[i]['task_name'], 'user':'AUTO'}
                if one_timestamp:
                    new_event['end_timestamp'] = time
                else:
                    new_event['timestamp'] = time
                    new_event['event_type'] = 'complete'
                aligned_trace.append(new_event)
                if not one_timestamp:
                    aligned_trace.append(dict(caseid=caseid,
                                              task=optimal_alignment[i]['task_name'],
                                              event_type='start',
                                              user='AUTO',
                                              timestamp=time))
            # If the event appears in the Log but not in the model delete the event from the trace
            elif movement_type =='L':
                j += 1
    elif alignment_data['fitness'] == 1:
        aligned_trace=raw_trace
    return aligned_trace

def trace_verification(aligned_trace, trace):
    """
    This method performs the completeness check,
    error correction and joints the start and complete events
    """
    tasks = list({x['task'] for x in aligned_trace})
    start_list = list(filter(lambda x: x['event_type'] == 'start', aligned_trace))
    complete_list = list(filter(lambda x: x['event_type'] == 'complete', aligned_trace))
    new_trace = list()
    missalignment = False
    for task in tasks:
        missalignment = False
        start_events = sorted(list(filter(lambda x: x['task'] == task, start_list)), key=itemgetter('timestamp'))
        complete_events = sorted(list(filter(lambda x: x['task'] == task, complete_list)), key=itemgetter('timestamp'))
        if(len(start_events) == len(complete_events)):
            for i, _ in enumerate(start_events):
                new_trace.append(dict(caseid=start_events[i]['caseid'],
                                 task=start_events[i]['task'],
                                 user=start_events[i]['user'],
                                 start_timestamp=start_events[i]['timestamp'],
                                 end_timestamp=complete_events[i]['timestamp']))
        else:
            missalignment = True
            break
    if not missalignment:
        new_trace = sorted(new_trace, key=itemgetter('start_timestamp'))
    else:
        new_trace = list()
    return new_trace

def append_start_end(trace, one_timestamp):
    for new_event in ['Start', 'End']:
        idx = 0 if new_event == 'Start' else -1
        t_key = 'end_timestamp'
        if not one_timestamp and new_event == 'Start':
            t_key = 'start_timestamp'
        temp_event = dict()
        temp_event['caseid'] = trace[idx]['caseid']
        temp_event['task'] = new_event
        temp_event['user'] = new_event
        time = trace[idx][t_key] + datetime.timedelta(microseconds=1)
        if new_event == 'Start':
            time = trace[idx][t_key] - datetime.timedelta(microseconds=1)
        temp_event['end_timestamp'] = time
        if not one_timestamp:
            temp_event['start_timestamp'] = time
        if new_event == 'Start':
            trace.insert(0, temp_event)
        else:
            trace.append(temp_event)
    return trace

# =============================================================================
# External tool calling
# =============================================================================

def evaluate_alignment(settings):
    """Evaluate business process traces alignment in relation with BPMN structure.
    Args:
        settings (dict): Path to jar and file names
    """
    print(" -- Evaluating event log alignment --")
    file_name = settings['file'].split('.')[0]
    args = ['java', '-jar', settings['align_path'],
            settings['output']+os.sep,
            file_name+'.xes',
            settings['file'].split('.')[0]+'.bpmn',
            'true']
    subprocess.call(args, bufsize=-1)

# --support --
# Methods for the reading of the alignment data generated by the proconformance plug-in
def read_alignment_info(filename):
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
                    optimal_alignment.append(dict(movement_type=result.group(1), task_name=result.group(3).strip()))
            records.append(dict(trace_type=trace_type,optimal_alignment=optimal_alignment))
    return records

def traces_alignment_type(filename):
    records = list()
    with open(filename) as fp:
        [next(fp) for i in range(7)]
        for line in fp:
            temp_record = line.split(',')
            records.append(dict(caseid=temp_record[2],trace_type=int(temp_record[1]),fitness=float(temp_record[11])))
    return records

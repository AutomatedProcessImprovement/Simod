# -*- coding: utf-8 -*-
import re
from support_modules import support as sup
from operator import itemgetter
import subprocess
import os

def align_traces(log, settings):
    """this method is the kernel of all the alignment process"""
    evaluate_alignment(settings)
    optimal_alignments = read_alignment_info(settings['aligninfo'])
    traces_alignments = traces_alignment_type(settings['aligntype'])
    raw_traces=log.get_raw_traces()
    aligned_traces = list()
    i = 0
    size = len(raw_traces)
    for raw_trace in raw_traces:
        try:
            # Alignment of each trace
            aligned_trace = process_trace(raw_trace, optimal_alignments, traces_alignments )
            # Conformity check and reformating
            aligned_trace = trace_verification(aligned_trace, raw_trace)
            if aligned_trace:
                aligned_traces.extend(aligned_trace)
        except Exception as e:
            print(str(e))
        sup.print_progress(((i / (size-1))* 100),'Aligning log traces with model ')
        i += 1
    sup.print_done_task()
    return aligned_traces

def process_trace(raw_trace, optimal_alignments, traces_alignments ):
    """this method performs the alignment of each trace according with the data optimal alignment"""
    caseid = raw_trace[0]['caseid']
    alignment_data = list(filter(lambda x: x['caseid'] == caseid, traces_alignments))[0]
    aligned_trace = list()
    # If fitness is 1 all the trace es aligned
    if alignment_data['fitness'] < 1 and alignment_data['fitness'] > 0:
        optimal_alignment = list(filter(lambda x: alignment_data['trace_type'] == x['trace_type'], optimal_alignments))[0]['optimal_alignment']
        j = 0
        for i in range(0,len(optimal_alignment)):
            movement_type = optimal_alignment[i]['movement_type']
            # If the Model and the log are aligned copy the raw value
            if movement_type =='LMGOOD':
                aligned_trace.append(raw_trace[j])
                j += 1
            # If the Log needs an extra task, create the start and complet event with time 0 and user AUTO
            elif movement_type =='MREAL':
                if i == 0:
                    time = raw_trace[i]['start_timestamp']
                else:
                    time = aligned_trace[-1]['start_timestamp']
                aligned_trace.append(dict(caseid=caseid, task=optimal_alignment[i]['task_name'], event_type='start', user='AUTO', start_timestamp=time, end_timestamp=time))
                aligned_trace.append(dict(caseid=caseid, task=optimal_alignment[i]['task_name'], event_type='complete', user='AUTO', start_timestamp=time, end_timestamp=time))
            # If the event appears in the Log but not in the model delete the event from the trace
            elif movement_type =='L':
                j += 1
    elif alignment_data['fitness'] == 1:
        aligned_trace=raw_trace
    return aligned_trace

def trace_verification(aligned_trace, raw_trace):
    """this method performs the conformity check, error correction and joints the start and complete events"""
    tasks = list({x['task'] for x in aligned_trace})
    start_list = list(filter(lambda x: x['event_type'] == 'start', aligned_trace))
    complete_list = list(filter(lambda x: x['event_type'] == 'complete', aligned_trace))
    new_trace = list()
    missalignment = False    
    for task in tasks:
        missalignment = False
        start_events = sorted(list(filter(lambda x: x['task'] == task, start_list)), key=itemgetter('start_timestamp'))
        complete_events = sorted(list(filter(lambda x: x['task'] == task, complete_list)), key=itemgetter('start_timestamp'))
        if(len(start_events) == len(complete_events)):
            for i, _ in enumerate(start_events):
                new_trace.append(dict(caseid=start_events[i]['caseid'], 
                                 task=start_events[i]['task'],
                                 event_type=start_events[i]['task'],
                                 user=start_events[i]['user'],
                                 start_timestamp=start_events[i]['start_timestamp'],
                                 end_timestamp=complete_events[i]['start_timestamp']))
        else:
            missalignment = True
            break
    if not missalignment:
        new_trace = sorted(new_trace, key=itemgetter('start_timestamp'))
    else:
        new_trace = list()
    return new_trace

# =============================================================================
# External tool calling
# =============================================================================

def evaluate_alignment(settings):
    """Evaluate business process traces alignment in relation with BPMN structure.
    Args:
        settings (dict): Path to jar and file names
    """
    print(" -- Evaluating event log alignment --")
    args = ['java', '-jar', settings['align_path'],
            settings['output']+os.sep,
            settings['file'],
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

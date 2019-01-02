# -*- coding: utf-8 -*-
from readers import bpmn_reader as br
from readers import log_reader as lr
from extraction import process_structure as gph
from extraction import log_replayer2 as rpl

import support as sup
import os
import sys, getopt

def read_inputs(start_timeformat, end_timeformat, log_columns_numbers, log_file_name, bpmn_file_name, ns_include=True):
    # Reading and parsing of config file
    log, bpmn = None, None
    try:
        log = lr.LogReader(log_file_name, log_columns_numbers, start_timeformat, end_timeformat, ns_include)
        bpmn = br.BpmnReader(bpmn_file_name)
    except IOError as e:
        print('Input error ' + str(e))
    except Exception as e:
        print('Unexpected error...' + '\n' + str(e))
    return log, bpmn

def import_bimp_statistics(output_dir, bpmn_file_name, source='bimp'):
    sim_statistics = list()
    for root, dirs, files in os.walk(output_dir):
        for i in range(0, len(files)):
            f = files[i]
            timeformat = '%Y-%m-%d %H:%M:%S.%f'
            log , bpmn = read_inputs(timeformat, timeformat, [0,1,1,4,2,3], os.path.join(output_dir,f), bpmn_file_name)
            # Creation of process graph
            process_graph = gph.create_process_structure(bpmn)
            # Process replaying
            conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log, source=source, run_num=i)
            sim_statistics.extend(process_stats)
    [x.update(dict(role=x['resource'])) for x in sim_statistics]
    return sim_statistics

def import_scylla_statistics(output_dir, file_name, bpmn_file_name, parameters, run_num=0, source='scylla'):
    sim_statistics = list()
    timeformat = '%Y-%m-%dT%H:%M:%S.000'
    log , bpmn = read_inputs(timeformat, timeformat, [], os.path.join(output_dir ,file_name), bpmn_file_name, False)
    # Creation of process graph
    process_graph = gph.create_process_structure(bpmn)
    # Process replaying
    conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log, source=source, run_num=run_num)
    sim_statistics.extend(process_stats)
    for stat in sim_statistics:
        resource = stat['resource'].split('#')[0][:-1]
        role = list(filter(lambda x: x['id'] == resource, parameters['resource_pool']))[0]['name']
        stat.update(dict(role=role))
    return sim_statistics

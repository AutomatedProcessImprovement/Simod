# -*- coding: utf-8 -*-
import os

from support_modules.readers import log_reader as lr
from support_modules.readers import bpmn_reader as br
from extraction import process_structure as gph
from extraction import log_replayer as rpl


def read_inputs(timeformat, log_columns_numbers, log_file_name, bpmn_file_name, ns_include=True):
    # Reading and parsing of config file
    log, bpmn = None, None
    try:
        log = lr.LogReader(log_file_name, timeformat, timeformat, log_columns_numbers, ns_include)
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
            log , bpmn = read_inputs(timeformat, [0,1,1,4,2,3], os.path.join(output_dir,f), bpmn_file_name)
            # Creation of process graph
            
            process_graph = gph.create_process_structure(bpmn)
            # Process replaying
            conformed_traces, not_conformed_traces, process_stats = rpl.replay(process_graph, log, source=source, run_num=i)
            sim_statistics.extend(process_stats)
    return sim_statistics
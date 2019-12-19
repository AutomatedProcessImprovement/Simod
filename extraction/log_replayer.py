# -*- coding: utf-8 -*-
import networkx as nx
from support_modules import support as sup

from collections import OrderedDict

# settings['read_options']['one_timestamp']
def replay(process_graph, log, settings, source='log', run_num=0):
    subsec_set = create_subsec_set(process_graph)
    parallel_gt_exec = parallel_execution_list(process_graph)
    not_conformant_traces = list()
    conformant_traces=list()
    process_stats=list()
    traces = log.get_traces()
    for index in range(0,len(traces)):
        trace_times = list()
        trace = traces[index]
        temp_gt_exec = parallel_gt_exec
        cursor = list()
        current_node = find_task_node(process_graph,trace[0]['task'])
        cursor.append(current_node)
        removal_allowed = True
        is_conformant = True
        #----time recording------
        trace_times.append(create_record(trace, 0))
        #------------------------
        for i in range(1, len(trace)):
            next_node = find_task_node(process_graph,trace[i]['task'])
            # If loop management
            if next_node == cursor[-1]:
                prev_record = find_previous_record(trace_times, process_graph.node[next_node]['name'])
                trace_times.append(create_record(trace, i, prev_record))
                process_graph.node[next_node]['executions'] += 1
            else:
                try:
                    cursor, prev_node = update_cursor(next_node, process_graph, cursor)
                    #----time recording------
                    prev_record = find_previous_record(trace_times, process_graph.node[prev_node]['name'])
                    trace_times.append(create_record(trace, i, prev_record))
                    process_graph.node[next_node]['executions'] += 1
                    #------------------------
                except:
                    is_conformant = False
                    break
                for element in reversed(cursor[:-1]):
                    # Process AND
                    if process_graph.node[element]['type'] == 'gate3':
                        gate = [d for d in temp_gt_exec if d['nod_num'] == element][0]
                        gate.update(dict(executed= gate['executed'] + 1))
                        if gate['executed'] < gate['num_paths']:
                            removal_allowed = False
                        else:
                            removal_allowed = True
                            cursor.remove(element)
                    # Process Task
                    elif process_graph.node[element]['type'] == 'task':
                        if (element,next_node) in subsec_set:
                            if removal_allowed:
                                cursor.remove(element)
                    # Process other
                    else:
                        if removal_allowed:
                            cursor.remove(element)
        if not is_conformant:
            not_conformant_traces.append(trace)
        else:
            conformant_traces.append(trace)
            process_stats.extend(trace_times)
        sup.print_progress(((index / (len(traces)-1))* 100),'Replaying process traces ')
    #------Filtering records and calculate stats---
    process_stats = list(filter(lambda x: x['task'] != 'Start' and x['task'] != 'End' and x['resource'] != 'AUTO', process_stats))
    process_stats = calculate_process_metrics(process_stats)
    [x.update(dict(source=source, run_num=run_num)) for x in process_stats]
    #----------------------------------------------
    sup.print_done_task()
    #------conformance percentage------------------
#    print('Conformance percentage: ' + str(sup.ffloat((len(conformant_traces)/len(traces)) * 100,2)) + '%')
    #----------------------------------------------
    return conformant_traces, not_conformant_traces, process_stats

def update_cursor(nnode,process_graph,cursor):
    tasks = list(filter(lambda x: process_graph.node[x]['type']=='task',cursor))
    shortest_path = list()
    prev_node = 0
    for pnode in reversed(tasks):
        try:
            shortest_path = list(nx.shortest_path(process_graph, pnode, nnode))[1:]
            prev_node = pnode
            break
        except nx.NetworkXNoPath:
            pass
    if len(list(filter(lambda x: process_graph.node[x]['type']=='task',shortest_path))) > 1:
        raise Exception('Incoherent path')
    ap_list = cursor + shortest_path
    # Preserve order and leave only new
    cursor = list(OrderedDict.fromkeys(ap_list))
    return cursor, prev_node

def create_record(trace, index, one_timestamp, last_event=dict()):
    
    start_time = trace[index]['start_timestamp']
    end_time = trace[index]['end_timestamp']
    caseid = trace[index]['caseid']
    resource = trace[index]['user']
    if not bool(last_event):
        enabling_time = trace[index]['end_timestamp']
    else:
        enabling_time = last_event['end_timestamp']
    return dict(caseid=caseid,task=trace[index]['task'],start_timestamp=start_time,
        end_timestamp=end_time,enable_timestamp=enabling_time,resource=resource)

def find_previous_record(trace_times, task):
    event = dict()
    for x in trace_times[::-1]:
        if task == x['task']:
            event = x
            break
    return event

def calculate_process_metrics(process_stats):
    for record in process_stats:
        duration=(record['end_timestamp']-record['start_timestamp']).total_seconds()
        waiting=(record['start_timestamp']-record['enable_timestamp']).total_seconds()
        multitasking=0
        #TODO check resourse for multi_tasking
        if waiting<0:
            waiting=0
            if record['end_timestamp'] > record['enable_timestamp']:
                duration=(record['end_timestamp']-record['enable_timestamp']).total_seconds()
                multitasking=(record['enable_timestamp']-record['start_timestamp']).total_seconds()
            else:
                multitasking = duration
        record['processing_time'] = duration
        record['waiting_time'] = waiting
        record['multitasking'] = multitasking
    return process_stats

def create_subsec_set(process_graph):
    subsec_set = set()
    task_list = list(filter(lambda x: process_graph.node[x]['type']=='task' , list(nx.nodes(process_graph))))
    for task in task_list:
        next_tasks = sup.reduce_list(find_next_tasks(process_graph, task))
        for n_task in next_tasks:
            subsec_set.add((task,n_task))
    return subsec_set

def parallel_execution_list(process_graph):
    execution_list = list()
    para_gates = list(filter(lambda x: process_graph.node[x]['type'] =='gate3',nx.nodes(process_graph)))
    for x in para_gates:
        execution_list.append(dict(nod_num=x, num_paths=len(list(process_graph.neighbors(x))), executed=0))
    return execution_list

def find_next_tasks(process_graph, num):
    tasks_list=list()
    for node in process_graph.neighbors(num):
        if process_graph.node[node]['type']=='task' or process_graph.node[node]['type']=='start' or process_graph.node[node]['type']=='end':
            tasks_list.append([node])
        else:
            tasks_list.append(find_next_tasks(process_graph, node))
    return 	tasks_list

def find_task_node(process_graph,task_name):
    resp = list(filter(lambda x: process_graph.node[x]['name'] == task_name ,process_graph.nodes))
    if len(resp)>0:
        resp = resp[0]
    else:
        raise Exception('Task not found on bpmn structure...')
    return resp

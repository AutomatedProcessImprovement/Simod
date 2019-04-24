# -*- coding: utf-8 -*-
import networkx as nx
from support_modules import support as sup

from collections import OrderedDict

def replay(process_graph, traces):
    subsec_set = create_subsec_set(process_graph)
    parallel_gt_exec = parallel_execution_list(process_graph)
    not_conformant_traces = list()
    conformant_traces=list()
    for index in range(0,len(traces)):
        trace = traces[index]
        temp_gt_exec = parallel_gt_exec
        cursor = list()
        current_node = find_task_node(process_graph,trace[0]['task'])
        cursor.append(current_node)
        removal_allowed = True
        is_conformant = True
        for i in range(1, len(trace)):
            next_node = find_task_node(process_graph,trace[i]['task'])
            # If loop management
            if next_node == cursor[-1]:
                process_graph.node[next_node]['executions'] += 1
            else:
                try:
                    cursor, prev_node = update_cursor(next_node, process_graph, cursor)
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
        sup.print_progress(((index / (len(traces)-1))* 100),'Replaying process traces ')
    sup.print_done_task()
    return conformant_traces, not_conformant_traces

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

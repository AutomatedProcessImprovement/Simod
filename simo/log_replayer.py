# -*- coding: utf-8 -*-
import networkx as nx
import support as sup
import datetime
import support as sup

from datetime import date
from operator import itemgetter


def replay(process_graph, log):
    traces = log.get_traces()
    not_conformed=list()
    conformed=list()
    process_stats=list()
    # index = 0
    # for i in range(0,len(traces)):
    #     if traces[i][0]['caseid']=='Case93':
    #         index = i
    # for i in range(index, index+1):
    for i in range(0, len(traces)):
        trace=traces[i]
        trace_number = trace[0]['caseid']
        # print('---' + str(trace_number))
        reload_temp_values(process_graph)
        # Load start and end times on graph
        load_time_in_nodes(process_graph, trace)
        # Check conformity
        # [print(x['task']) for x in trace ]
        not_conformed, conformed, is_conform = check_conformity(process_graph,trace,not_conformed, conformed,trace_number)
        if is_conform:
            process_stats.append(calculate_stats(process_graph))
        sup.print_progress(((i / (len(traces)-1))* 100),'Replaying process traces ')
        # Filtering traces
    conformed_traces = filter_traces(traces, conformed)
    not_conformed_traces = filter_traces(traces, not_conformed)
    sup.print_done_task()
    return conformed_traces, not_conformed_traces, process_stats

#TODO documentar
def check_conformity(process_graph,trace,not_conformed, conformed, trace_number):
    is_conform=True
    start_node = find_task_node(process_graph, trace[0]['task'])
    process_graph.node[start_node]['temp_enable']=process_graph.node[start_node]['temp_start']
    process_graph.node[start_node]['tsk_act']= True
    for i in range(1,len(trace)):
        # print('------')
        event = trace[i]
        current_node = find_task_node(process_graph, event['task'])
        try:
            # Enabling task
            if current_node != find_task_node(process_graph, trace[i-1]['task']):
                prev_node = find_enabled_task(process_graph, current_node)
                # print(process_graph.node[prev_node]['name'])
                # print(process_graph.node[current_node]['name'])
                path = list(nx.shortest_simple_paths(process_graph, prev_node, current_node))
                if len(path)==1:
                    path = path[0]
                else:
                    path = choose_path(process_graph,path)
                process_path(process_graph,path)
            else:
                prev_node = current_node
                # print(process_graph.node[prev_node]['name'])
                # print(process_graph.node[current_node]['name'])
            process_graph.node[current_node]['temp_enable']=process_graph.node[prev_node]['temp_end']
        except Exception as e:
            # print(str(e))
            is_conform=False
            not_conformed.append(trace_number)
            break
    if is_conform and check_parallel_gateways_execution(process_graph):
        if i == len(trace) -1:
            conformed.append(trace_number)
        else:
            not_conformed.append(trace_number)
            is_conform = False
    return not_conformed, conformed, is_conform

def process_path(process_graph,path):
    is_and_gate = False
    for i in range(1,len(path)-1):
        node = path[i]
        if(process_graph.node[node]['type']=='gate3'):
            if not (process_graph.node[node]['gtact']):
                process_graph.node[node]['gtact']=True
            process_graph.node[node]['gt_visited_paths']+=1
            if process_graph.node[node]['gt_num_paths'] <= process_graph.node[node]['gt_visited_paths']:
                process_graph.node[path[0]]['tsk_act']=False
            process_graph.node[path[-1]]['tsk_act']=True
            is_and_gate = True
            break
    if not is_and_gate:
        process_graph.node[path[-1]]['tsk_act']=True
        process_graph.node[path[0]]['tsk_act']=False

def find_enabled_task(process_graph, num):
    prev_tasks = sup.reduce_list(find_prev_tasks(process_graph, num))
    prev_exec_tasks = list(filter( lambda x: process_graph.node[x]['tsk_act'] == True, prev_tasks))
    #regular case
    if len(prev_exec_tasks) == 1:
        return prev_exec_tasks[0]
    # case parallel gate and multiple task enabled return the one with the last time
    elif len(prev_exec_tasks) > 1:
        temp=process_graph.node[prev_exec_tasks[0]]['temp_end']
        node = prev_exec_tasks[0]
        for t_node in prev_exec_tasks:
            if temp < process_graph.node[t_node]['temp_end']:
                temp = process_graph.node[t_node]['temp_end']
                node = t_node
        return node
    else:
        raise Exception('No enabled task...')

def choose_path(process_graph,paths):
    temp_paths = paths
    for temp_path in temp_paths:
        j=1
        incoherent = False
        while (j < len(temp_path)-1) and not incoherent:
            if(process_graph.node[temp_path[j]]['type']=='task'):
                incoherent = True
            j +=1
        if incoherent == True:
            temp_paths.remove(temp_path)
    if len(temp_paths)==1:
        return temp_paths[0]
    elif len(paths)>1:
        temp=temp_paths[0]
        for path in temp_paths:
            if len(path) < len(temp):
                temp = path
        return temp
    else:
        raise Exception('Incoherent path... intermediate tasks without execute')



def filter_traces(traces, criteria):
    response = list()
    for case in criteria:
        for trace in traces:
            if case == trace[0]['caseid']:
                response.append(trace)
    return response

def load_time_in_nodes(process_graph, trace):
    for event in trace:
        try:
            current_node = find_task_node(process_graph, event['task'])
        except Exception as e:
            print(str(e))
        else:
            process_graph.node[current_node]['temp_start'] = event['start_timestamp']
            process_graph.node[current_node]['temp_end'] = event['end_timestamp']

def define_max_time(process_graph, tasks):
    if len(tasks)>=1:
        max_node = tasks[0]
        for task in tasks:
            if process_graph.node[task]['temp_end'] > process_graph.node[max_node]['temp_end']:
                max_node = task
    else:
        #TODO check why this happend
        raise Exception('not enabling tasks')
    return max_node

def reload_temp_values(process_graph):
    for node_number in nx.nodes(process_graph):
        process_graph.node[node_number]['temp_enable']=None
        process_graph.node[node_number]['temp_start']=None
        process_graph.node[node_number]['temp_end']=None
        process_graph.node[node_number]['gtact']=False
        process_graph.node[node_number]['xor_gtdir']=0
        process_graph.node[node_number]['gt_visited_paths']=0
        process_graph.node[node_number]['tsk_act']=False


def print_times(process_graph):
    executed_nodes = list(filter( lambda x: not process_graph.node[x]['temp_start'] == None, nx.nodes(process_graph)))
    sorted_info = list()
    for node_number in executed_nodes:
        sorted_info.append(dict(name=process_graph.node[node_number]['name'],enabled=process_graph.node[node_number]['temp_enable'],
        start=process_graph.node[node_number]['temp_start'],end=process_graph.node[node_number]['temp_end'],
        processing_times=process_graph.node[node_number]['processing_times'] ,waiting_times= process_graph.node[node_number]['waiting_times'],
        executions= process_graph.node[node_number]['executions']))
    sorted_info = sorted(sorted_info,key=itemgetter('enabled'))
    for info in sorted_info:
        print('-------')
        print(info['name'])
        print('temp_enable: '+ str(info['enabled']))
        print('temp_start: '+ str(info['start']))
        print('temp_end: '+ str(info['end']))
        print('temp_start: '+ str(info['start']))
        print('processing_times: '+ str(info['processing_times']))
        print('waiting_times: '+ str(info['waiting_times']))
        print('executions: '+ str(info['executions']))


def find_prev_tasks(process_graph, num):
    r_process_graph = process_graph.reverse(copy=True)
    tasks_list=list()
    for node in r_process_graph.neighbors(num):
        if r_process_graph.node[node]['type']=='task' or r_process_graph.node[node]['type']=='start' or r_process_graph.node[node]['type']=='end':
            tasks_list.append([node])
        else:
            tasks_list.append(find_prev_tasks(process_graph, node))
    return 	tasks_list

def find_next_tasks(process_graph, num):
    tasks_list=list()
    for node in process_graph.neighbors(num):
        if process_graph.node[node]['type']=='task' or process_graph.node[node]['type']=='start' or process_graph.node[node]['type']=='end':
            tasks_list.append([node])
        else:
            tasks_list.append(find_prev_tasks(process_graph, node))
    return 	tasks_list

def find_task_node(process_graph,task_name):
    resp = list(filter(lambda x: process_graph.node[x]['name'] == task_name ,process_graph.nodes))
    if len(resp)>0:
        resp = resp[0]
    else:
        raise Exception('Task not found on bpmn structure...')
    return resp

#TODO documentar
def calculate_stats(process_graph):
    executed_nodes = list(filter( lambda x: not process_graph.node[x]['temp_start'] == None, nx.nodes(process_graph)))
    total_processing,total_waiting,total_multitasking = 0,0,0
    for node_number in executed_nodes:
        duration=(process_graph.node[node_number]['temp_end']-process_graph.node[node_number]['temp_start']).total_seconds()
        waiting=(process_graph.node[node_number]['temp_start']-process_graph.node[node_number]['temp_enable']).total_seconds()
        multitasking=0
        #TODO check resourse for multi_tasking
        if waiting<0:
            waiting=0
            if process_graph.node[node_number]['temp_end'] > process_graph.node[node_number]['temp_enable']:
                duration=(process_graph.node[node_number]['temp_end']-process_graph.node[node_number]['temp_enable']).total_seconds()
                multitasking=(process_graph.node[node_number]['temp_enable']-process_graph.node[node_number]['temp_start']).total_seconds()
            else:
                multitasking = duration
        process_graph.node[node_number]['multi_tasking'].append(multitasking)
        process_graph.node[node_number]['processing_times'].append(duration)
        process_graph.node[node_number]['waiting_times'].append(waiting)
        process_graph.node[node_number]['executions'] += 1
        total_processing+=duration
        total_waiting+=waiting
        total_multitasking+=multitasking
    return dict(total_processing=total_processing, total_waiting=total_waiting, total_multitasking=total_multitasking)

def check_parallel_gateways_execution(process_graph):
    completed = True
    para_gates = list(filter(lambda x: (process_graph.node[x]['type'] =='gate3') and (process_graph.node[x]['gtact'] ==True),nx.nodes(process_graph)))
    for x in para_gates:
        if process_graph.node[x]['gt_num_paths']!= process_graph.node[x]['gt_visited_paths']:
            completed = False
            break
    return completed

# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from .. import support_utils as sup


def create_process_structure(bpmn, drawing=False, verbose=True):
    # Loading of bpmn structure into a directed graph
    g = load_process_structure(bpmn, verbose)
    if drawing:
        graph_network_x(g)
    if verbose:
        sup.print_done_task()
    return g


def graph_network_x(g):
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, with_labels=True)
    plt.draw()
    plt.show()


def find_node_num(g, id):
    resp = list(filter(lambda x: g.nodes[x]['id'] == id, g.nodes))
    if len(resp) > 0:
        resp = resp[0]
    else:
        resp = -1
    return resp


def create_nodes(g, total_elements, index, array, node_type, node_name, node_id, verbose):
    i = 0
    while i<len(array):
        if verbose:
            sup.print_progress(((index / (total_elements-1))* 100),'Loading of bpmn structure from file ')
        g.add_node(index,type=node_type,name=array[i][node_name],id=array[i][node_id],
            executions=0, processing_times=list(), waiting_times=list(), multi_tasking=list(),
            temp_enable=None, temp_start=None, temp_end=None,tsk_act=False,
            gtact=False, xor_gtdir=0, gt_num_paths=0, gt_visited_paths=0)
        index +=1
        i +=1
    return index


def load_process_structure(bpmn, verbose):
    g = nx.DiGraph()
    # Loading data
    start = bpmn.get_start_event_info()
    tasks = bpmn.get_tasks_info()
    ex_gates = bpmn.get_ex_gates_info()
    inc_gates = bpmn.get_inc_gates_info()
    para_gates = bpmn.get_para_gates_info()
    end = bpmn.get_end_event_info()
    timer_events = bpmn.get_timer_events_info()
    # total_elements = (len(start) + len(tasks) + len(ex_gates) + len(inc_gates) + len(para_gates) + len(end) + len(timer_events))
    total_elements = (len(tasks) + len(ex_gates) + len(inc_gates) + len(para_gates) + len(timer_events))
    #Adding nodes
    # index = create_nodes(g,total_elements,0,start,'start','start_name','start_id')
    index = create_nodes(g, total_elements, 0, list(filter(lambda x: x['task_name']=='Start',tasks)),'start','task_name','task_id', verbose)
    index = create_nodes(g, total_elements, index, list(filter(lambda x: x['task_name'] not in ['Start', 'End'],tasks)),'task','task_name','task_id', verbose)
    index = create_nodes(g, total_elements, index, list(filter(lambda x: x['task_name']=='End',tasks)),'end','task_name','task_id', verbose)
    index = create_nodes(g, total_elements, index, list(filter(lambda x: x['gate_dir']=='Diverging',ex_gates)),'gate','gate_name','gate_id', verbose)
    index = create_nodes(g, total_elements, index, list(filter(lambda x: x['gate_dir']=='Converging',ex_gates)),'gate2','gate_name','gate_id', verbose)
    index = create_nodes(g, total_elements, index, inc_gates,'gate2','gate_name','gate_id', verbose)
    index = create_nodes(g, total_elements, index, para_gates,'gate3','gate_name','gate_id', verbose)
    # index = create_nodes(g, total_elements, index, end,'end','end_name','end_id')
    index = create_nodes(g, total_elements, index, timer_events,'timer','timer_name','timer_id', verbose)
    # Add edges
    for edge in bpmn.get_edges_info():
        if edge['source'] != start[0]['start_id'] and edge['target'] != end[0]['end_id']:
            g.add_edge(find_node_num(g, edge['source']) ,find_node_num(g, edge['target']))
    # Define #of in_paths for paralell gateways_probabilities
    para_gates = list(filter(lambda x: g.nodes[x]['type'] =='gate3',nx.nodes(g)))
    for x in para_gates:
        g.nodes[x]['gt_num_paths'] = len(list(g.neighbors(x)))
    return g

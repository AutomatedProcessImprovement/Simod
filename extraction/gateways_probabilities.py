# -*- coding: utf-8 -*-
from support_modules import support as sup
import numpy as np

def define_probabilities(process_graph,bpmn,log, type):
    # Analisys of gateways probabilities
    if (type==1):
        gateways = analize_gateways(process_graph,log)
    elif(type==2):
        gateways = analize_gateways_random(process_graph,log)
    elif(type==3):
        gateways = analize_gateways_equi(process_graph,log)
    # Creating response list
    response = list()
    gateways=normalize_probabilities(process_graph,gateways)
    for gateway in gateways:
        # print("gateway prob", process_graph.node[gateway['gate']]['id'])
        gatewayId = process_graph.node[gateway['gate']]['id']
        for path in gateway['targets']:
            sequence_id = bpmn.find_sequence_id(process_graph.node[gateway['gate']]['id'],process_graph.node[path['out_node']]['id'])
            response.append(dict(gatewayid=gatewayId,elementid=sequence_id,prob=path['probability']))
    sup.print_done_task()
    return response

def normalize_probabilities(process_graph,gateways):
    for gateway in gateways:
        probabilities = list()
        for path in gateway['targets']:
            probabilities.append(path['probability'])
        probabilities = sup.round_preserve(probabilities,1)
        for i in range(0, len(probabilities)):
            gateway['targets'][i]['probability'] = probabilities[i]
    return gateways

def extract_target_tasks(process_graph, num):
    tasks_list=list()
    for node in process_graph.neighbors(num):
        if process_graph.node[node]['type']=='task' or process_graph.node[node]['type']=='start' or process_graph.node[node]['type']=='end':
            tasks_list.append([node])
        else:
            tasks_list.append(extract_target_tasks(process_graph, node))
    return     tasks_list

def analize_gateway_structure(process_graph, gate_num):
    # Sources
    r = process_graph.reverse(copy=True)
    paths = list(r.neighbors(gate_num))
    task_paths = extract_target_tasks(r, gate_num)
    in_paths = [sup.reduce_list(path) for path in task_paths]
    ins = [dict(in_tasks=y, in_node= x) for x,y in zip(paths, in_paths)]

    # Targets
    paths = list(process_graph.neighbors(gate_num))
    task_paths = extract_target_tasks(process_graph, gate_num)
    out_paths = [sup.reduce_list(path) for path in task_paths]
    outs = [dict(out_tasks=y, out_node= x, ocurrences=0, probability=0) for x,y in zip(paths, out_paths)]

    return dict(gate=gate_num,sources=ins,targets=outs)

def analize_gateways(process_graph,log):
    nodes_list = list()
    for node in process_graph.nodes:
        if process_graph.node[node]['type']=='gate':
            nodes_list.append(analize_gateway_structure(process_graph,node))

    i=0
    for node in nodes_list:
        if len(nodes_list) > 1:
            sup.print_progress(((i / (len(nodes_list)-1))* 100),'Analysing gateways probabilities ')
        else:
            sup.print_progress(((i / (len(nodes_list)))* 100),'Analysing gateways probabilities ')

        total_ocurrences = 0
        for path in node['targets']:
            ocurrences = 0
            for out_task in path['out_tasks']:
                ocurrences += process_graph.node[out_task]['executions']
            path['ocurrences'] = ocurrences
            total_ocurrences += path['ocurrences']
        for path in node['targets']:
            if total_ocurrences > 0:
                probability = path['ocurrences']/total_ocurrences
#                print(node['gate'],process_graph.node[path['out_node']]['name'], path['ocurrences'],probability, sep=' ')
            else:
                probability = 0
            path['probability'] = round(probability,2)
        i+=1
    return nodes_list

def analize_gateways_random(process_graph,log):
    nodes_list = list()
    for node in process_graph.nodes:
        if process_graph.node[node]['type']=='gate':
            nodes_list.append(analize_gateway_structure(process_graph,node))
    i=0
    for node in nodes_list:
        sup.print_progress(((i / (len(nodes_list)-1))* 100),'Analysing gateways random probabilities ')
        prob_array = np.random.dirichlet(np.ones(len(node['targets'])),size=1)
        for i in range(0,len(node['targets'])):
            node['targets'][i]['probability'] = round(float(prob_array[0][i]),2)
        i+=1
    return nodes_list

def analize_gateways_equi(process_graph,log):
    nodes_list = list()
    for node in process_graph.nodes:
        if process_graph.node[node]['type']=='gate':
            nodes_list.append(analize_gateway_structure(process_graph,node))
    i=0
    for node in nodes_list:
        sup.print_progress(((i / (len(nodes_list)-1))* 100),'Analysing gateways random probabilities ')
        p = 1/len(node['targets'])
        prob_array = list()
        [prob_array.append(p) for i in range(0,len(node['targets'])) ]
        for i in range(0,len(node['targets'])):
            node['targets'][i]['probability'] = round(float(prob_array[i]),2)
        i+=1
    return nodes_list

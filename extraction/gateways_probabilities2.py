# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:56:37 2020

@author: Manuel Camargo
"""

# -*- coding: utf-8 -*-
#%%
from support_modules import support as sup
import numpy as np

#%% Methods

class GatewaysEvaluator():
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, process_graph, method):
        """constructor"""
        self.process_graph = process_graph
        self.method = method
        self.probabilities = list()
        self.define_probabilities()


    def define_probabilities(self):
        # Analisys of gateways probabilities
        if self.method=='discovery':
            gateways = self.analize_gateways()
        elif self.method=='random':
            gateways = self.analize_gateways_random()
        elif self.method=='equiprobable':
            gateways = self.analize_gateways_equi()
        # Creating response list
        gateways = self.normalize_probabilities(gateways)
        for gateway in gateways:
            gatewayId = self.process_graph.node[gateway['gate']]['id']
            for path in gateway['targets']:
                # sequence_id = bpmn.find_sequence_id(gatewayId,
                #                                     self.process_graph.node[path['out_node']]['id'])
                self.probabilities.append(
                    dict(gatewayid=gatewayId,
                         out_path_id=self.process_graph.node[path['out_node']]['id'],
                         prob=path['probability']))
                # response.append(dict(gatewayid=gatewayId,
                #                      elementid=sequence_id,prob=path['probability']))
        sup.print_done_task()

    
    def normalize_probabilities(self, gateways):
        for gateway in gateways:
            probabilities = list()
            print(gateway)
            for path in gateway['targets']:
                probabilities.append(path['probability'])
            probabilities = sup.round_preserve(probabilities, 1)
            probabilities = sup.avoid_zero_prob(probabilities)
            for i in range(0, len(probabilities)):
                gateway['targets'][i]['probability'] = probabilities[i]
        return gateways
    
    def extract_target_tasks(self, graph, num):
        tasks_list=list()
        for node in graph.neighbors(num):
            if graph.node[node]['type'] in ['task', 'start', 'end']:
                tasks_list.append([node])
            else:
                tasks_list.append(self.extract_target_tasks(graph, node))
        return tasks_list
    
    def analize_gateway_structure(self, gate_num):
        # Sources
        r = self.process_graph.reverse(copy=True)
        paths = list(r.neighbors(gate_num))
        task_paths = self.extract_target_tasks(r, gate_num)
        in_paths = [sup.reduce_list(path) for path in task_paths]
        ins = [dict(in_tasks=y, in_node= x) for x, y in zip(paths, in_paths)]
    
        # Targets
        paths = list(self.process_graph.neighbors(gate_num))
        task_paths = self.extract_target_tasks(self.process_graph, gate_num)
        out_paths = [sup.reduce_list(path) for path in task_paths]
        outs = [dict(out_tasks=y, out_node= x, ocurrences=0, probability=0)
                for x, y in zip(paths, out_paths)]
    
        return dict(gate=gate_num,sources=ins,targets=outs)
    
    def analize_gateways(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            if self.process_graph.node[node]['type']=='gate':
                nodes_list.append(self.analize_gateway_structure(node))
    
        i=0
        for node in nodes_list:
            if len(nodes_list) > 1:
                sup.print_progress(((i / (len(nodes_list)-1))* 100),
                                   'Analysing gateways probabilities ')
            else:
                sup.print_progress(((i / (len(nodes_list)))* 100),
                                   'Analysing gateways probabilities ')
    
            total_ocurrences = 0
            for path in node['targets']:
                ocurrences = 0
                for out_task in path['out_tasks']:
                    ocurrences += self.process_graph.node[out_task]['executions']
                path['ocurrences'] = ocurrences
                total_ocurrences += path['ocurrences']
            for path in node['targets']:
                if total_ocurrences > 0:
                    probability = path['ocurrences']/total_ocurrences
                else:
                    probability = 0
                path['probability'] = round(probability,2)
            i+=1
        return nodes_list
    
    def analize_gateways_random(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            if self.process_graph.node[node]['type']=='gate':
                nodes_list.append(self.analize_gateway_structure(node))
        i=0
        for node in nodes_list:
            sup.print_progress(((i / (len(nodes_list)-1))* 100),
                               'Analysing gateways random probabilities ')
            prob_array = np.random.dirichlet(np.ones(len(node['targets'])), 
                                             size=1)
            for i in range(0,len(node['targets'])):
                node['targets'][i]['probability'] = round(
                    float(prob_array[0][i]),2)
            i+=1
        return nodes_list
    
    def analize_gateways_equi(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            if self.process_graph.node[node]['type']=='gate':
                nodes_list.append(self.analize_gateway_structure(node))
        i=0
        for node in nodes_list:
            sup.print_progress(((i / (len(nodes_list)-1))* 100),
                               'Analysing gateways random probabilities ')
            p = 1/len(node['targets'])
            prob_array = list()
            [prob_array.append(p) for i in range(0,len(node['targets'])) ]
            for i in range(0,len(node['targets'])):
                node['targets'][i]['probability'] = round(
                    float(prob_array[i]),2)
            i+=1
        return nodes_list

#%%
import json
import networkx as nx
from networkx.readwrite import json_graph
#%%
   
with open('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_simo/process_graph.json') as file:
    gdata = json.load(file)
    file.close()

process_graph = json_graph.node_link_graph(gdata)

with open('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_simo/id.json') as file:
    id_data = json.load(file)
    file.close()
id_data = {int(k):v for k, v in id_data.items()}
nx.set_node_attributes(process_graph, id_data)
gevaluator = GatewaysEvaluator(process_graph, 'discovery')
sequences = gevaluator.probabilities
print(sequences)

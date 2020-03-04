# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:56:37 2020

@author: Manuel Camargo
"""

# -*- coding: utf-8 -*-
#%%
# from support_modules import support as sup
import support as sup

import numpy as np
import itertools
import pandas as pd
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
        # Fix 0 probabilities and float error sums
        gateways = self.normalize_probabilities(gateways)
        # Creating response list
        gids = lambda x: self.process_graph.node[x['gate']]['id']
        gateways['gatewayid'] = gateways.apply(gids, axis=1)
        gids = lambda x: self.process_graph.node[x['t_path']]['id']
        gateways['out_path_id'] = gateways.apply(gids, axis=1)
        self.probabilities = gateways[['gatewayid',
                                       'out_path_id',
                                       'probability']].to_dict('records')
        sup.print_done_task()

    def normalize_probabilities(self, nodes_list):
        temp_list = list()
        for key, group in nodes_list.groupby(by=['gate']):
            probabilities = group.probability.tolist()
            probabilities = sup.round_preserve(probabilities, 1)
            probabilities = sup.avoid_zero_prob(probabilities)
            for i in range(0, len(probabilities)):
                group.iat[i, 3] = probabilities[i]
            temp_list.extend(group.to_dict('records'))
        return pd.DataFrame.from_records(temp_list)

    def extract_target_tasks(self, graph, num):
        tasks_list=list()
        for node in graph.neighbors(num):
            if graph.node[node]['type'] in ['task', 'start', 'end']:
                tasks_list.append([node])
            else:
                tasks_list.append(self.extract_target_tasks(graph, node))
        return tasks_list

    def analize_gateway_structure(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            outs = list()
            if self.process_graph.node[node]['type']=='gate':
                # Targets
                paths = list(self.process_graph.neighbors(node))
                task_paths = self.extract_target_tasks(self.process_graph, node)
                task_paths = [sup.reduce_list(path) for path in task_paths]
                for path, tasks in zip(paths, task_paths):
                    for task in tasks:
                        outs.append((node, path, task))
            nodes_list.extend(outs)
        gateways = pd.DataFrame.from_records(
            nodes_list,columns=['gate', 't_path', 't_task'])
        return gateways

    def analize_gateways(self):
        # Obtain gateways structure
        nodes_list = self.analize_gateway_structure()
        # Add task execution count
        executions = lambda x: self.process_graph.node[x['t_task']]['executions']
        nodes_list['executions'] = nodes_list.apply(executions, axis=1)
        # Aggregate path executions
        nodes_list = (nodes_list.groupby(by=['gate', 't_path'])['executions']
                      .sum()
                      .reset_index())
        # Calculate probabilities
        total_ocurrences = (nodes_list.groupby(by=['gate'])['executions']
                            .sum().to_dict())
        rate = lambda x: round(
            np.divide(x['executions'], total_ocurrences[x['gate']]), 2)
        nodes_list['probability'] = nodes_list.apply(rate, axis=1)
        return nodes_list

    def analize_gateways_random(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            if self.process_graph.node[node]['type']=='gate':
                nodes_list.append(self.analize_gateway_structure(node))
        i=0
        for node in nodes_list:
            sup.print_progress(((i / (len(nodes_list)-1))* 100),'Analysing gateways random probabilities ')
            prob_array = np.random.dirichlet(np.ones(len(node['targets'])),size=1)
            for i in range(0,len(node['targets'])):
                node['targets'][i]['probability'] = round(float(prob_array[0][i]),2)
            i+=1
        return nodes_list

    def analize_gateways_equi(self):
        nodes_list = list()
        for node in self.process_graph.nodes:
            if self.process_graph.node[node]['type']=='gate':
                nodes_list.append(self.analize_gateway_structure(node))
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

#%%
import json
import networkx as nx
from networkx.readwrite import json_graph
#%%

with open('C:/Users/Manuel Camargo/Documents/GitHub/SiMo-Discoverer/process_graph.json') as file:
    gdata = json.load(file)
    file.close()

process_graph = json_graph.node_link_graph(gdata)

with open('C:/Users/Manuel Camargo/Documents/GitHub/SiMo-Discoverer/id.json') as file:
    id_data = json.load(file)
    file.close()
id_data = {int(k):v for k, v in id_data.items()}
nx.set_node_attributes(process_graph, id_data)
gevaluator = GatewaysEvaluator(process_graph, 'discovery')
sequences = gevaluator.probabilities
print(sequences)

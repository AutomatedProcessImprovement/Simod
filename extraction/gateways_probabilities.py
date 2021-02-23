# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:56:37 2020

@author: Manuel Camargo
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import utils.support as sup
from tqdm import tqdm 


class GatewaysEvaluator():
    """
        This class evaluates the gateways' probabilities according with
        an spcified method defined by the user
     """

    def __init__(self, process_graph, method):
        """constructor"""
        self.process_graph = process_graph
        self.method = method
        self.probabilities = list()
        self.define_probabilities()

    def define_probabilities(self) -> None:
        """
        Defines the gateways' probabilities according with an spcified method

        """
        # sup.print_performed_task('Analysing gateways` probabilities')
        # Analisys of gateways probabilities
        if self.method == 'discovery':
            gateways = self.analize_gateways()
        elif self.method == 'random':
            gateways = self.analize_gateways_random()
        elif self.method == 'equiprobable':
            gateways = self.analize_gateways_equi()
        # Fix 0 probabilities and float error sums
        gateways = self.normalize_probabilities(gateways)
        # Creating response list
        gids = lambda x: self.process_graph.nodes[x['gate']]['id']
        gateways['gatewayid'] = gateways.apply(gids, axis=1)
        gids = lambda x: self.process_graph.nodes[x['t_path']]['id']
        gateways['out_path_id'] = gateways.apply(gids, axis=1)
        self.probabilities = gateways[['gatewayid',
                                       'out_path_id',
                                       'prob']].to_dict('records')
        # sup.print_done_task()

    @staticmethod
    def normalize_probabilities(nodes_list: pd.DataFrame) -> pd.DataFrame:
        temp_list = list()
        for key, group in nodes_list.groupby(by=['gate']):
            probabilities = np.nan_to_num(
                np.array(group.prob.tolist())).tolist()
            probabilities = sup.round_preserve(probabilities, 1)
            probabilities = sup.avoid_zero_prob(probabilities)
            for i in range(0, len(probabilities)):
                group.iat[i, 3] = probabilities[i]
            temp_list.extend(group.to_dict('records'))
        return pd.DataFrame.from_records(temp_list)

    def analize_gateway_structure(self) -> pd.DataFrame:
        """
        Creates a structure that contains the subsequent tasks of each
        gateway's paths

        Returns
        -------
        Dataframe
        """
        # look for source and target tasks
        def extract_target_tasks(graph: object, num: int) -> list:
            tasks_list = list()
            for node in graph.neighbors(num):
                if graph.nodes[node]['type'] in ['task', 'start', 'end']:
                    tasks_list.append([node])
                else:
                    tasks_list.append(extract_target_tasks(graph, node))
            return tasks_list

        nodes_list = list()
        for node in tqdm(self.process_graph.nodes, 
                         desc='analysing gateways probabilities:'):
            outs = list()
            if self.process_graph.nodes[node]['type'] == 'gate':
                # Targets
                paths = list(self.process_graph.neighbors(node))
                task_paths = extract_target_tasks(self.process_graph, node)
                task_paths = [sup.reduce_list(path) for path in task_paths]
                for path, tasks in zip(paths, task_paths):
                    for task in tasks:
                        outs.append((node, path, task))
            nodes_list.extend(outs)
        gateways = pd.DataFrame.from_records(
            nodes_list, columns=['gate', 't_path', 't_task'])
        return gateways

    def analize_gateways(self) -> pd.DataFrame:
        """
        Discovers the gateway's paths probabilities accordig with the
        historical information

        Returns
        -------
        nodes_list : DataFrame

        """
        # Obtain gateways structure
        nodes_list = self.analize_gateway_structure()
        # Add task execution count
        executions = lambda x: self.process_graph.nodes[x['t_task']]['executions']
        nodes_list['executions'] = nodes_list.apply(executions, axis=1)
        # Aggregate path executions
        nodes_list = (nodes_list.groupby(by=['gate', 't_path'])['executions']
                      .sum()
                      .reset_index())
        # Calculate probabilities
        t_ocurrences = (nodes_list.groupby(by=['gate'])['executions']
                        .sum().to_dict())
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = lambda x: round(
                np.divide(x['executions'], t_ocurrences[x['gate']]), 2)
            nodes_list['prob'] = nodes_list.apply(rate, axis=1)
        return nodes_list

    def analize_gateways_random(self) -> pd.DataFrame:
        """
        Assigns random probabilities to the gateway's paths

        Returns
        -------
        DataFrame
        """
        # Obtain gateways structure
        nodes_list = self.analize_gateway_structure()
        # Aggregate paths
        nodes_list = (nodes_list.groupby(by=['gate', 't_path'])
                      .count()
                      .reset_index())
        nodes_list['prob'] = 0.0
        # assign random probabilities
        temp_list = list()
        for key, group in nodes_list.groupby(by=['gate']):
            probabilities = np.random.dirichlet(np.ones(len(group)), size=1)[0]
            for i in range(0, len(probabilities)):
                group.iat[i, 3] = round(probabilities[i], 2)
            temp_list.extend(group.to_dict('records'))
        return pd.DataFrame.from_records(temp_list)

    def analize_gateways_equi(self) -> pd.DataFrame:
        """
        Assigns deterministic probabilities to the gateway's paths,
        the value is equiprobable for each path

        Returns
        -------
        DataFrame
        """
        # Obtain gateways structure
        nodes_list = self.analize_gateway_structure()
        # Aggregate paths
        nodes_list = (nodes_list.groupby(by=['gate', 't_path'])
                      .count()
                      .reset_index())
        nodes_list['prob'] = 0.0
        # assign probabilities
        temp_list = list()
        for key, group in nodes_list.groupby(by=['gate']):
            p = 1/len(group)
            probabilities = [p for i in range(0, len(group))]
            for i in range(0, len(probabilities)):
                group.iat[i, 3] = round(probabilities[i], 2)
            temp_list.extend(group.to_dict('records'))
        return pd.DataFrame.from_records(temp_list)

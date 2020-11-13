# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:47:09 2020

@author: Manuel Camargo
"""

from extraction import log_replayer as rpl
from extraction import interarrival_definition as arr
from extraction import gateways_probabilities as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch
from extraction import tasks_evaluator as te

import pandas as pd
import itertools
import utils.support as sup


class ParameterMiner():
    """
    This class extracts all the BPS parameters
    """

    def __init__(self, log, bpmn, process_graph, settings):
        """constructor"""
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        self.process_stats = list()
        self.parameters = dict()
        self.conformant_traces = list()
        self.resource_table = pd.DataFrame()

    def extract_parameters(self, num_inst, start_time) -> None:
        """
        main method for parameters extraction
        """
        self.replay_process()
        self.mine_resources()
        self.mine_interarrival()
        self.mine_gateways_probabilities()
        self.process_tasks()
        # TODO: Num of test partition
        self.parameters['instances'] = num_inst
        self.parameters['start_time'] = start_time

    def replay_process(self) -> None:
        """
        Process replaying
        """
        replayer = rpl.LogReplayer(self.process_graph, self.log.get_traces(),
                                   self.settings)
        self.process_stats = replayer.process_stats
        self.conformant_traces = replayer.conformant_traces

    def mine_resources(self) -> None:
        """
        Analysing resource pool LV917 or 247
        """
        res_analyzer = rl.ResourcePoolAnalyser(
            self.log,
            sim_threshold=self.settings['rp_similarity'])
        ttcreator = sch.TimeTablesCreator(self.settings)
        args = {'res_cal_met': self.settings['res_cal_met'], 
                'arr_cal_met': self.settings['arr_cal_met'], 
                'resource_table': res_analyzer.resource_table}
        ttcreator.create_timetables(args)
        resource_pool = self.create_resource_pool(res_analyzer.resource_table,
                                                  ttcreator.res_ttable_name)
        self.parameters['resource_pool'] = resource_pool
        self.parameters['time_table'] = ttcreator.time_table
        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        # resource_table.to_csv('pools_'+self.settings['file'].split('.')[0]+'.csv')
        self.process_stats = self.process_stats.merge(resource_table,
                                                      on='resource',
                                                      how='left')
        self.resource_table = resource_table

    def mine_interarrival(self) -> None:
        """
        Calculates the inter-arrival rate
        """
        inter_evaluator = arr.InterArrivalEvaluator(self.process_graph,
                                                    self.conformant_traces,
                                                    self.settings)
        self.parameters['arrival_rate'] = inter_evaluator.dist

    def mine_gateways_probabilities(self) -> None:
        """
        Gateways probabilities 1=Historical, 2=Random, 3=Equiprobable
        """
        gevaluator = gt.GatewaysEvaluator(self.process_graph,
                                          self.settings['gate_management'])
        sequences = gevaluator.probabilities
        for seq in sequences:
            seq['elementid'] = self.bpmn.find_sequence_id(seq['gatewayid'],
                                                          seq['out_path_id'])
        self.parameters['sequences'] = sequences

    def process_tasks(self) -> None:
        """
        Tasks id information
        """
        tevaluator = te.TaskEvaluator(self.process_graph,
                                      self.process_stats,
                                      self.parameters['resource_pool'],
                                      self.settings)
        self.parameters['elements_data'] = tevaluator.elements_data
        
    @staticmethod       
    def create_resource_pool(resource_table, table_name) -> list():
        """
        Creates resource pools and associate them the default timetable
        in BIMP format
        """
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM',
                              'total_amount': '20', 'costxhour': '20',
                              'timetable_id': table_name['arrival']}]
        data = sorted(resource_table, key=lambda x: x['role'])
        for key, group in itertools.groupby(data, key=lambda x: x['role']):
            res_group = [x['resource'] for x in list(group)]
            r_pool_size = str(len(res_group))
            name = (table_name['resources'] if 'resources' in table_name.keys()
                    else table_name[key])
            resource_pool.append({'id': sup.gen_id(),
                                  'name': key,
                                  'total_amount': r_pool_size,
                                  'costxhour': '20',
                                  'timetable_id': name})
        return resource_pool

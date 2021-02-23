# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:47:09 2020

@author: Manuel Camargo
"""
import pandas as pd
import itertools
import utils.support as sup
import traceback

from extraction import log_replayer as rpl
from extraction import interarrival_definition as arr
from extraction import gateways_probabilities as gt
from extraction import role_discovery as rl
from extraction import schedule_tables as sch
from extraction import tasks_evaluator as te


class StructureParametersMiner():
    """
    This class extracts all the BPS parameters
    """
    class Decorators(object):

        @classmethod
        def safe_exec(cls, method):
            """
            Decorator to safe execute methods and return the state
            ----------
            method : Any method.
            Returns
            -------
            dict : execution status
            """
            def safety_check(*args, **kw):
                is_safe = kw.get('is_safe', method.__name__.upper())
                if is_safe:
                    try:
                        method(*args)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        is_safe = False
                return is_safe
            return safety_check

    def __init__(self, log, bpmn, process_graph, settings):
        """constructor"""
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        # inter-arrival times and durations by default mean an exponential
        # 'manual', 'automatic', 'semi-automatic', 'default'
        self.settings['pdef_method'] = 'default'
        # self.settings['rp_similarity'] = 0.5
        self.process_stats = list()
        self.parameters = dict()
        self.conformant_traces = list()
        self.is_safe = True

    def extract_parameters(self, num_inst, start_time, resource_pool) -> None:
        """
        main method for parameters extraction
        """
        self.is_safe = self._replay_process(is_safe=self.is_safe)
        self.is_safe = self._mine_interarrival(is_safe=self.is_safe)
        self.is_safe = self._mine_gateways_probabilities(is_safe=self.is_safe)
        self.is_safe = self._process_tasks(resource_pool, is_safe=self.is_safe)

        # TODO: Num of test partition
        self.parameters['instances'] = num_inst
        self.parameters['start_time'] = start_time

    @Decorators.safe_exec
    def _replay_process(self) -> None:
        """
        Process replaying
        """
        replayer = rpl.LogReplayer(self.process_graph,
                                   self.log.get_traces(),
                                   self.settings,
                                   msg='reading conformant training traces:')
        self.process_stats = replayer.process_stats
        self.process_stats['role'] = 'SYSTEM'
        self.conformant_traces = replayer.conformant_traces

    # @Decorators.safe_exec
    @staticmethod
    def mine_resources(settings, log) -> None:
        """
        Analysing resource pool LV917 or 247
        """
        parameters = dict()
        settings['res_cal_met'] = 'default'
        settings['res_dtype'] = '247'  # 'LV917', '247'
        settings['arr_cal_met'] = 'default'
        settings['arr_dtype'] = '247'  # 'LV917', '247'
        ttcreator = sch.TimeTablesCreator(settings)
        args = {'res_cal_met': settings['res_cal_met'],
                'arr_cal_met': settings['arr_cal_met']}
        ttcreator.create_timetables(args)
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM',
                          'total_amount': '100000', 'costxhour': '20',
                          'timetable_id': ttcreator.res_ttable_name['arrival']}]

        parameters['resource_pool'] = resource_pool
        parameters['time_table'] = ttcreator.time_table
        return parameters

    @Decorators.safe_exec
    def _mine_interarrival(self) -> None:
        """
        Calculates the inter-arrival rate
        """
        inter_evaluator = arr.InterArrivalEvaluator(self.process_graph,
                                                    self.conformant_traces,
                                                    self.settings)
        self.parameters['arrival_rate'] = inter_evaluator.dist

    @Decorators.safe_exec
    def _mine_gateways_probabilities(self) -> None:
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

    @Decorators.safe_exec
    def _process_tasks(self, resource_pool,) -> None:
        """
        Tasks id information
        """
        tevaluator = te.TaskEvaluator(self.process_graph,
                                      self.process_stats,
                                      resource_pool,
                                      self.settings)
        self.parameters['elements_data'] = tevaluator.elements_data

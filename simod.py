# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
import subprocess
import types
import copy

import pandas as pd
import numpy as np
from operator import itemgetter

from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
import xmltodict as xtd
from lxml import etree

import utils.support as sup
from utils.support import timeit
import readers.log_reader as lr
import readers.bpmn_reader as br
import readers.process_structure as gph
import readers.log_splitter as ls
from support_modules.writers import xml_writer as xml
from support_modules.writers import xes_writer as xes
from support_modules.analyzers import sim_evaluator as sim
from support_modules.log_repairing import conformance_checking as chk

from extraction import parameter_extraction as par
from extraction import log_replayer as rpl


class Simod():
    """
    Main class of the Simulation Models Discoverer
    """

    def __init__(self, settings):
        """constructor"""
        self.status = STATUS_OK
        self.settings = settings

        self.log = types.SimpleNamespace()
        self.log_train = types.SimpleNamespace()
        self.log_test = types.SimpleNamespace()
        self.bpmn = types.SimpleNamespace()
        self.process_graph = types.SimpleNamespace()

        self.sim_values = list()
        self.response = dict()
        self.parameters = dict()

    def execute_pipeline(self, mode, can=False) -> None:
        exec_times = dict()
        if mode in ['optimizer']:
            self.temp_path_redef()
        if self.status == STATUS_OK:
            self.read_inputs(log_time=exec_times)
        if self.status == STATUS_OK:
            self.evaluate_alignment(log_time=exec_times)
        if self.status == STATUS_OK:
            self.extract_parameters(log_time=exec_times)
        if self.status == STATUS_OK:
            self.simulate(log_time=exec_times)
        self.mannage_results()
        if self.status == STATUS_OK:
            self.save_times(exec_times, self.settings)
        if self.status == STATUS_OK and can == True:
            self.export_canonical_model()
        print("-- End of trial --")

    def temp_path_redef(self) -> None:
        # Paths redefinition
        self.settings['output'] = os.path.join('outputs', sup.folder_id())
        if self.settings['alg_manag'] == 'repair':
            try:
                self.settings['aligninfo'] = os.path.join(
                    self.settings['output'],
                    'CaseTypeAlignmentResults.csv')
                self.settings['aligntype'] = os.path.join(
                    self.settings['output'],
                    'AlignmentStatistics.csv')
            except Exception as e:
                print(e)
                self.status = STATUS_FAIL

    @timeit
    def read_inputs(self, **kwargs) -> None:
        # Output folder creation
        if not os.path.exists(self.settings['output']):
            os.makedirs(self.settings['output'])
            os.makedirs(os.path.join(self.settings['output'], 'sim_data'))
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['input'],
                                             self.settings['file']),
                                self.settings['read_options'])
        # Time splitting 80-20
        self.split_timeline(0.8,
                            self.settings['read_options']['one_timestamp'])
        # Create customized event-log for the external tools
        xes.XesWriter(self.log, self.settings)
        # Execution steps
        self.mining_structure(self.settings)
        self.bpmn = br.BpmnReader(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'.bpmn'))
        self.process_graph = gph.create_process_structure(self.bpmn)
        # Replaying test partition
        print("-- Reading test partition --")
        try:
            test_replayer = rpl.LogReplayer(
                self.process_graph,
                self.get_traces(self.log_test,
                                self.settings['read_options']['one_timestamp']),
                self.settings)
            self.process_stats = test_replayer.process_stats
            self.process_stats = pd.DataFrame.from_records(self.process_stats)
            self.log_test = test_replayer.conformant_traces
        except AssertionError as e:
            print(e)
            self.status = STATUS_FAIL
            print("-- End of trial --")
            

    @timeit
    def evaluate_alignment(self, **kwargs) -> None:
        """
        Evaluate alignment
        """
        # Evaluate alignment
        try:
            chk.evaluate_alignment(self.process_graph,
                                   self.log_train,
                                   self.settings)
        except Exception as e:
            print(e)
            self.status = STATUS_FAIL

    @timeit
    def extract_parameters(self, **kwargs) -> None:
        print("-- Mining Simulation Parameters --")
        try:
            p_extractor = par.ParameterMiner(self.log_train,
                                             self.bpmn,
                                             self.process_graph,
                                             self.settings)
            num_inst = len(pd.DataFrame(self.log_test).caseid.unique())
            start_time = (pd.DataFrame(self.log_test)
                          .start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
            p_extractor.extract_parameters(num_inst, start_time)
            self.process_stats = self.process_stats.merge(
                p_extractor.resource_table[['resource', 'role']],
                on='resource',
                how='left')
            # save parameters
            self.parameters = copy.deepcopy(p_extractor.parameters)
            self.parameters['rol_user'] = p_extractor.resource_table
            # print parameters in xml bimp format
            xml.print_parameters(os.path.join(
                self.settings['output'],
                self.settings['file'].split('.')[0]+'.bpmn'),
                os.path.join(self.settings['output'],
                             self.settings['file'].split('.')[0]+'.bpmn'),
                p_extractor.parameters)
        except Exception as e:
            print(e)
            self.status = STATUS_FAIL

    @timeit
    def simulate(self, **kwargs) -> None:
        for rep in range(self.settings['repetitions']):
            print("Experiment #" + str(rep + 1))
            try:
                self.execute_simulator(self.settings, rep)
                self.process_stats = self.process_stats.append(
                    self.read_stats(self.settings, self.bpmn, rep),
                    ignore_index=True,
                    sort=False)
                evaluator = sim.SimilarityEvaluator(
                    self.process_stats,
                    self.settings,
                    rep)
                metrics = [self.settings['sim_metric']]
                if 'add_metrics' in self.settings.keys():
                    metrics = list(set(list(self.settings['add_metrics']) +
                                       metrics))
                for metric in metrics:
                    evaluator.measure_distance(metric)
                    self.sim_values.append(evaluator.similarity)
            except Exception as e:
                print(e)
                self.status = STATUS_FAIL
                break

    def mannage_results(self) -> None:
        self.response, measurements = self.define_response(self.status,
                                                           self.sim_values,
                                                           self.settings)
        if self.settings['exec_mode'] in ['optimizer'] and measurements:
            if os.path.getsize(os.path.join('outputs',
                                            self.settings['temp_file'])) > 0:
                sup.create_csv_file(measurements,
                                    os.path.join('outputs',
                                                 self.settings['temp_file']),
                                    mode='a')
            else:
                sup.create_csv_file_header(measurements,
                                           os.path.join(
                                               'outputs',
                                               self.settings['temp_file']))
        elif self.settings['exec_mode'] == 'single':
            print('------ Final results ------')
            [print(k, v, sep=': ') for k, v in self.response.items()
             if k != 'params']
            self.response.pop('params', None)
            if measurements:
                sup.create_csv_file_header(measurements,
                                           os.path.join(
                                               'outputs',
                                               self.settings['temp_file']))

    @staticmethod
    def define_response(status, sim_values, settings):
        response = dict()
        measurements = list()
        data = {'alg_manag': settings['alg_manag'],
                'epsilon': settings['epsilon'],
                'eta': settings['eta'],
                'rp_similarity': settings['rp_similarity'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
        if settings['res_cal_met'] in ['discovered' ,'pool']:
            data['res_cal_met'] = settings['res_cal_met']
            data['res_support'] = settings['res_support']
            data['res_confidence'] = settings['res_confidence']
        if settings['arr_cal_met'] == 'discovered':
            data['arr_support'] = settings['arr_support']
            data['arr_confidence'] = settings['arr_confidence']
        if settings['exec_mode'] == 'optimizer':
            similarity = 0
            response['params'] = settings
            if status == STATUS_OK:
                similarity = np.mean([x['sim_val'] for x in sim_values
                                      if x['metric'] == settings['sim_metric']])
                loss = ((1 - similarity) if not settings['sim_metric'] in ['mae', 'log_mae']
                        else similarity)
                response['loss'] = loss
                response['status'] = status if loss > 0 else STATUS_FAIL
                for sim_val in sim_values:
                    measurements.append({
                        **{'similarity': sim_val['sim_val'],
                           'sim_metric': sim_val['metric'],
                           'status': response['status']},
                        **data})
            else:
                response['status'] = status
                measurements.append({
                    **{'similarity': 0, 
                       'sim_metric': settings['sim_metric'],
                       'status': response['status']},
                    **data})
        else:
            if status == STATUS_OK:
                similarity = np.mean([x['sim_val'] for x in sim_values
                                      if x['metric'] == settings['sim_metric']])
                response['similarity'] = similarity
                response['params'] = settings
                response['status'] = status if similarity > 0 else STATUS_FAIL
                response = {**response, **data}
                for sim_val in sim_values:
                    measurements.append({
                        **{'similarity': sim_val['sim_val'],
                           'sim_metric': sim_val['metric'],
                           'status': response['status']},
                        **data})
            else:
                response['similarity'] = 0
                response['sim_metric'] = settings['sim_metric']
                response['status'] = status
                response = {**response, **data}
        return response, measurements

    @staticmethod
    def save_times(times, settings):
        times = [{**{'output': settings['output']}, **times}]
        log_file = os.path.join('outputs', 'execution_times.csv')
        if not os.path.exists(log_file):
                open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)
            
    def export_canonical_model(self):
        ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
        time_table = etree.tostring(self.parameters['time_table'], pretty_print=True)
        time_table = xtd.parse(time_table, process_namespaces=True, namespaces=ns)
        self.parameters['time_table'] = time_table
        # Users in rol data
        user_rol = dict()
        for key, group in self.parameters['rol_user'].groupby('role'):
            user_rol[key] = list(group.resource)
        self.parameters['rol_user'] = user_rol
        # Json creation
        sup.create_json(self.parameters, os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'_canon.json'))
        

# =============================================================================
# Support methods
# =============================================================================

    def split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not
        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(self.log.data)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True).to_dict('records'))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))
        
    @staticmethod
    def get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces

# =============================================================================
# External tools calling
# =============================================================================
    @staticmethod
    def mining_structure(settings):
        """Execute splitminer for bpmn structure mining.
        Args:
            settings (dict): Path to jar and file names
            epsilon (double): Parallelism threshold (epsilon) in [0,1]
            eta (double): Percentile for frequency threshold (eta) in [0,1]
        """
        print(" -- Mining Process Structure --")
        # Event log file_name
        file_name = settings['file'].split('.')[0]
        input_route = os.path.join(settings['output'], file_name+'.xes')
        # Mining structure definition
        args = ['java', '-jar', settings['miner_path'],
                str(settings['epsilon']), str(settings['eta']), input_route,
                os.path.join(settings['output'], file_name)]
        subprocess.call(args)

    @staticmethod
    def execute_simulator(settings, rep):
        """Executes BIMP Simulations.
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        print("-- Executing BIMP Simulations --")
        args = ['java', '-jar', settings['bimp_path'],
                os.path.join(settings['output'],
                             settings['file'].split('.')[0]+'.bpmn'),
                '-csv',
                os.path.join(settings['output'], 'sim_data',
                             settings['file']
                             .split('.')[0]+'_'+str(rep+1)+'.csv')]
        subprocess.call(args)

    @staticmethod
    def read_stats(settings, bpmn, rep):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        m_settings = dict()
        m_settings['output'] = settings['output']
        m_settings['file'] = settings['file']
        column_names = {'resource': 'user'}
        m_settings['read_options'] = settings['read_options']
        m_settings['read_options']['timeformat'] = '%Y-%m-%d %H:%M:%S.%f'
        m_settings['read_options']['column_names'] = column_names
        temp = lr.LogReader(os.path.join(
            m_settings['output'], 'sim_data',
            m_settings['file'].split('.')[0] + '_'+str(rep + 1)+'.csv'),
            m_settings['read_options'])
        process_graph = gph.create_process_structure(bpmn)
        results_replayer = rpl.LogReplayer(process_graph,
                                           temp.get_traces(),
                                           settings,
                                           source='simulation',
                                           run_num=rep + 1)
        temp_stats = results_replayer.process_stats
        temp_stats['role'] = temp_stats['resource']
        return temp_stats


# =============================================================================
# Hyperparameter-optimizer
# =============================================================================


class DiscoveryOptimizer():
    """
    Hyperparameter-optimizer class
    """

    def __init__(self, settings, args, can=False):
        """constructor"""
        self.space = self.define_search_space(settings, args)
        self.settings = settings
        self.args = args
        self.can = can
        # Trials object to track progress
        self.bayes_trials = Trials()

    @staticmethod
    def define_search_space(settings, args):
        space = {**{'epsilon': hp.uniform('epsilon',
                                          args['epsilon'][0],
                                          args['epsilon'][1]),
                    'eta': hp.uniform('eta',
                                      args['eta'][0],
                                      args['eta'][1]),
                    'alg_manag': hp.choice('alg_manag',
                                           args['alg_manag']),
                    'rp_similarity': hp.uniform('rp_similarity',
                                                args['rp_similarity'][0],
                                                args['rp_similarity'][1]),
                    'gate_management': hp.choice('gate_management',
                                                 args['gate_management']),
                    'res_cal_met': hp.choice('res_cal_met',
                        [('discovered',{
                            'res_support': hp.uniform('res_support', 
                                                      args['res_sup_dis'][0], 
                                                      args['res_sup_dis'][1]),
                            'res_confidence': hp.uniform('res_confidence',
                                                         args['res_con_dis'][0],
                                                         args['res_con_dis'][1])}),
                          ('default', {
                              'res_dtype': hp.choice('res_dtype',
                                                     args['res_dtype'])})
                         ]),
                    'arr_cal_met': hp.choice('arr_cal_met',
                        [
                            ('discovered',{
                            'arr_support': hp.uniform('arr_support', 
                                                      args['arr_support'][0], 
                                                      args['arr_support'][1]),
                            'arr_confidence': hp.uniform('arr_confidence',
                                                         args['arr_confidence'][0],
                                                         args['arr_confidence'][1])}),
                          ('default', {
                              'arr_dtype': hp.choice('arr_dtype',
                                                     args['arr_dtype'])})
                         ])
                    },
                 **settings}
        return space

    def execute_trials(self):
        # create a new instance of Simod
        def exec_simod(instance_settings):
            # resources discovery
            method, values = instance_settings['res_cal_met']
            if method in 'discovered':
                instance_settings['res_confidence'] = values['res_confidence']
                instance_settings['res_support'] = values['res_support']
            else:
                instance_settings['res_dtype'] = values['res_dtype']
            instance_settings['res_cal_met'] = method
            # arrivals calendar
            method, values = instance_settings['arr_cal_met']
            if method in 'discovered':
                instance_settings['arr_confidence'] = values['arr_confidence']
                instance_settings['arr_support'] = values['arr_support']
            else:
                instance_settings['arr_dtype'] = values['arr_dtype']
            instance_settings['arr_cal_met'] = method
            simod = Simod(instance_settings)
            simod.execute_pipeline(self.settings['exec_mode'], can=self.can)
            return simod.response
        # Optimize
        
        best = fmin(fn=exec_simod,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.args['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        print('------ Final results ------')
        [print(k, v) for k, v in best.items()]


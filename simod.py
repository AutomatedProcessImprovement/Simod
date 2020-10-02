# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
import subprocess
import types
import itertools
import copy

import pandas as pd
import numpy as np
from operator import itemgetter

from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

from support_modules import support as sup
from support_modules.support import timeit
from support_modules.readers import log_reader as lr
from support_modules.readers import bpmn_reader as br
from support_modules.readers import process_structure as gph
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

    def execute_pipeline(self, mode) -> None:
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
        self.split_timeline(0.2,
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
        p_extractor = par.ParameterMiner(self.log_train,
                                         self.bpmn,
                                         self.process_graph,
                                         self.settings)
        num_inst = len(pd.DataFrame(self.log_test).caseid.unique())
        p_extractor.extract_parameters(num_inst)
        self.process_stats = self.process_stats.merge(
            p_extractor.resource_table[['resource', 'role']],
            on='resource',
            how='left')
        # print parameters in xml bimp format
        xml.print_parameters(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'.bpmn'),
            os.path.join(self.settings['output'],
                         self.settings['file'].split('.')[0]+'.bpmn'),
            p_extractor.parameters)

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
                evaluation = sim.SimilarityEvaluator(
                    self.process_stats,
                    self.settings,
                    rep,
                    metric=self.settings['sim_metric'])
                self.sim_values.append(evaluation.similarity)
            except Exception as e:
                print(e)
                self.status = STATUS_FAIL
                break

    def mannage_results(self) -> None:
        self.response, measurements = self.define_response(self.status,
                                                           self.sim_values,
                                                           self.settings)

        if measurements:
            if self.settings['exec_mode'] in ['optimizer']:
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
            else:
                print('------ Final results ------')
                [print(k, v, sep=': ') for k, v in self.response.items()
                 if k != 'params']
                self.response.pop('params', None)
                sup.create_csv_file_header([self.response],
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
        if settings['exec_mode'] == 'optimizer':
            similarity = 0
            response['params'] = settings
            if status == STATUS_OK:
                similarity = np.mean([x['sim_val'] for x in sim_values])
                loss = ((1 - similarity) if settings['sim_metric'] != 'mae'
                        else similarity)
                response['loss'] = loss
                response['status'] = status if loss > 0 else STATUS_FAIL
            else:
                response['status'] = status
            for sim_val in sim_values:
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                        'status': response['status']},
                    **data})
        else:
            if status == STATUS_OK:
                similarity = np.mean([x['sim_val'] for x in sim_values])
                response['similarity'] = similarity
                response['params'] = settings
                response['status'] = status if similarity > 0 else STATUS_FAIL
                response = {**response, **data}
            else:
                response['similarity'] = 0
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

# =============================================================================
# Support methods
# =============================================================================
    def split_timeline(self, percentage: float, one_timestamp: bool) -> None:
        """
        Split an event log dataframe to peform split-validation

        Parameters
        ----------
        percentage : float, validation percentage.
        one_timestamp : bool, Support only one timestamp.
        """
        # log = self.log.data.to_dict('records')
        self.log_train = copy.deepcopy(self.log)
        log = sorted(self.log_train.data, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_dict(log)
        log.sort_values(by='end_timestamp', ascending=False, inplace=True)

        num_events = int(np.round(len(log)*percentage))

        df_test = log.iloc[:num_events]
        df_train = log.iloc[num_events:]

        # Incomplete final traces
        df_train = df_train.sort_values(by=['caseid', 'pos_trace'],
                                        ascending=True)
        inc_traces = pd.DataFrame(df_train.groupby('caseid')
                                  .last()
                                  .reset_index())
        inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
        inc_traces = inc_traces['caseid'].to_list()

        # Drop incomplete traces
        df_test = df_test[~df_test.caseid.isin(inc_traces)]
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])

        df_train = df_train[~df_train.caseid.isin(inc_traces)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])

        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        self.log_test = (df_test
                         .sort_values(key, ascending=True)
                         .reset_index(drop=True).to_dict('records'))
        self.log_train.set_data(df_train
                                .sort_values(key, ascending=True)
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

    def __init__(self, settings, args):
        """constructor"""
        self.space = self.define_search_space(settings, args)
        self.settings = settings
        self.args = args
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
                                           ['replacement',
                                            'repair',
                                            'removal']),
                    'rp_similarity': hp.uniform('rp_similarity',
                                                args['rp_similarity'][0],
                                                args['rp_similarity'][1]),
                    'gate_management': hp.choice('gate_management',
                                                 args['gate_management'])},
                 **settings}
        return space

    def execute_trials(self):
        # create a new instance of Simod
        def exec_simod(instance_settings):
            simod = Simod(instance_settings)
            simod.execute_pipeline(self.settings['exec_mode'])
            return simod.response
        # Optimize
        best = fmin(fn=exec_simod,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.args['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        print(best)


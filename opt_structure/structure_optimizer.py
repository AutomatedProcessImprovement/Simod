# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:48:57 2020

@author: Manuel Camargo
"""
import os
import subprocess
import copy
import multiprocessing
from multiprocessing import Pool
import itertools
# import traceback
import numpy as np
import pandas as pd
from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

import utils.support as sup
from utils.support import timeit
import readers.log_splitter as ls
import readers.log_reader as lr
from support_modules.writers import xes_writer as xes
from support_modules.writers import xml_writer as xml
from support_modules.analyzers import sim_evaluator as sim

import opt_structure.structure_miner as sm
import opt_structure.structure_params_miner as spm
from tqdm import tqdm
import time


class StructureOptimizer():
    """
    Hyperparameter-optimizer class
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
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)
                    except Exception as e:
                        print(e)
                        # traceback.print_exc()
                        response['status'] = STATUS_FAIL
                return response
            return safety_check

    def __init__(self, settings, args, log):
        """constructor"""
        self.space = self.define_search_space(settings, args)
        # Read inputs
        self.log = log
        self._split_timeline(0.8, settings['read_options']['one_timestamp'])
        self.org_log = copy.deepcopy(log)
        self.org_log_train = copy.deepcopy(self.log_train)
        self.org_log_valdn = copy.deepcopy(self.log_valdn)
        # Load settings
        self.settings = settings
        self.file_name = os.path.join('outputs', sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        self.args = args
        # Trials object to track progress
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_parms = dict()

    @staticmethod
    def define_search_space(settings, args):
        var_dim = {'alg_manag': hp.choice('alg_manag',
                                           args['alg_manag']),
                    'gate_management': hp.choice('gate_management',
                                                 args['gate_management'])}
        if settings['mining_alg'] == 'sm1':
            var_dim['epsilon'] = hp.uniform('epsilon',
                                          args['epsilon'][0],
                                          args['epsilon'][1])
            var_dim['eta'] = hp.uniform('eta',
                                          args['eta'][0],
                                          args['eta'][1])
        elif settings['mining_alg'] == 'sm2':
            var_dim['concurrency'] = hp.uniform('concurrency',
                                          args['concurrency'][0],
                                          args['concurrency'][1])
        space = {**var_dim, **settings}
        return space

    def execute_trials(self):
        parameters, resource_table = (
            spm.StructureParametersMiner.mine_resources(
                self.settings, self.log_train))
        self.log_train = copy.deepcopy(self.org_log_train)
        def exec_pipeline(trial_stg):
            
            print('train split:', 
                  len(pd.DataFrame(self.log_train.data).caseid.unique()), 
                  ', valdn split:', 
                  len(pd.DataFrame(self.log_valdn).caseid.unique()),
                  sep=' ')
            # Vars initialization
            status = STATUS_OK
            exec_times = dict()
            sim_values = []
            # Path redefinition
            rsp = self._temp_path_redef(trial_stg,
                                        status=status,
                                        log_time=exec_times)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg
            # Structure mining
            rsp = self._mine_structure(trial_stg,
                                       status=status,
                                       log_time=exec_times)
            status = rsp['status']
            # Parameters extraction
            rsp = self._extract_parameters(trial_stg,
                                           rsp['values'],
                                           copy.deepcopy(parameters),
                                           resource_table,
                                           status=status,
                                           log_time=exec_times)
            status = rsp['status']
            # Simulate model
            rsp = self._simulate(trial_stg,
                                 self.log_valdn,
                                 status=status,
                                 log_time=exec_times)
            status = rsp['status']
            sim_values = rsp['values'] if status == STATUS_OK else sim_values
            # Save times
            self._save_times(exec_times, trial_stg)
            # Optimizer results
            rsp = self._define_response(trial_stg, status, sim_values)
            # reinstate log
            self.log = copy.deepcopy(self.org_log)
            self.log_train = copy.deepcopy(self.org_log_train)
            self.log_valdn = copy.deepcopy(self.org_log_valdn)
            print("-- End of trial --")
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.args['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        # Save results
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                       .sort_values('loss', ascending=bool))
            self.best_output = (results[results.status=='ok']
                                .head(1).iloc[0].output)
            self.best_parms = best
        except Exception as e:
            print(e)
            pass

    @timeit(rec_name='PATH_DEF')
    @Decorators.safe_exec
    def _temp_path_redef(self, settings, **kwargs) -> None:
        # Paths redefinition
        settings['output'] = os.path.join('outputs', sup.folder_id())
        if settings['alg_manag'] == 'repair':
            settings['aligninfo'] = os.path.join(
                settings['output'],
                'CaseTypeAlignmentResults.csv')
            settings['aligntype'] = os.path.join(
                settings['output'],
                'AlignmentStatistics.csv')
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
            os.makedirs(os.path.join(settings['output'], 'sim_data'))
        # Create customized event-log for the external tools
        xes.XesWriter(self.log, settings)
        return settings

    @timeit(rec_name='MINING_STRUCTURE')
    @Decorators.safe_exec
    def _mine_structure(self, settings, **kwargs) -> None:
        structure_miner = sm.StructureMiner(settings, self.log_train)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            return [structure_miner.bpmn, structure_miner.process_graph]
        else:
            raise RuntimeError('Mining Structure error')

    @timeit(rec_name='EXTRACTING_PARAMS')
    @Decorators.safe_exec
    def _extract_parameters(self, settings, structure, parameters, resource_table, **kwargs) -> None:
        bpmn, process_graph = structure
        p_extractor = spm.StructureParametersMiner(self.log_train,
                                                   bpmn,
                                                   process_graph,
                                                   resource_table,
                                                   settings)
        num_inst = len(self.log_valdn.caseid.unique())
        # Get minimum date
        start_time = (self.log_valdn
                      .start_timestamp
                      .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        p_extractor.extract_parameters(num_inst, start_time, parameters['resource_pool'])
        if p_extractor.is_safe:
            parameters = {**parameters, **p_extractor.parameters}
            # print parameters in xml bimp format
            xml.print_parameters(os.path.join(
                settings['output'],
                settings['file'].split('.')[0]+'.bpmn'),
                os.path.join(settings['output'],
                             settings['file'].split('.')[0]+'.bpmn'),
                parameters)

            self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
            self.log_valdn['source'] = 'log'
            self.log_valdn['run_num'] = 0
            self.log_valdn = self.log_valdn.merge(
                p_extractor.resource_table[['resource', 'role']],
                on='resource',
                how='left')
            self.log_valdn = self.log_valdn[
                ~self.log_valdn.task.isin(['Start', 'End'])]
        else:
            raise RuntimeError('Parameters extraction error')

    @timeit(rec_name='SIMULATION_EVAL')
    @Decorators.safe_exec
    def _simulate(self, settings, data,**kwargs) -> list:
        
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()
            
        reps = settings['repetitions']
        cpu_count = multiprocessing.cpu_count()
        w_count =  reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        args = [(settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')
        # Read simulated logs
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')
        # Evaluate
        args = [(settings, data, log) for log in p.get()]
        p = pool.map_async(self.evaluate_logs, args)
        pbar_async(p, 'evaluating results:')
        pool.close()
        # Save results
        sim_values = list(itertools.chain(*p.get()))
        return sim_values

    @staticmethod
    def read_stats(args):
        def read(settings, rep):
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
                m_settings['read_options'],
                verbose=False)
            temp = pd.DataFrame(temp.data)
            temp.rename(columns={'user': 'resource'}, inplace=True)
            temp['role'] = temp['resource']
            temp['source'] = 'simulation'
            temp['run_num'] = rep + 1
            temp = temp[~temp.task.isin(['Start', 'End'])]
            return temp
        return read(*args)

    @staticmethod
    def evaluate_logs(args):
        def evaluate(settings, data, sim_log):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            rep = (sim_log.iloc[0].run_num) - 1
            sim_values = list()
            data = data.append(
                sim_log,
                ignore_index=True,
                sort=False)
            evaluator = sim.SimilarityEvaluator(
                data,
                settings,
                rep)
            evaluator.measure_distance('dl')
            sim_values.append(evaluator.similarity)
            return sim_values
        return evaluate(*args)

    @staticmethod
    def execute_simulator(args):
        def sim_call(settings, rep):
            """Executes BIMP Simulations.
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            args = ['java', '-jar', settings['bimp_path'],
                    os.path.join(settings['output'],
                                  settings['file'].split('.')[0]+'.bpmn'),
                    '-csv',
                    os.path.join(settings['output'], 'sim_data',
                                  settings['file']
                                  .split('.')[0]+'_'+str(rep+1)+'.csv')]
            subprocess.run(args, check=True, stdout=subprocess.PIPE)
        sim_call(*args)

    @staticmethod
    def _save_times(times, settings):
        if times:
            times = [{**{'output': settings['output']}, **times}]
            log_file = os.path.join('outputs', 'execution_times.csv')
            if not os.path.exists(log_file):
                    open(log_file, 'w').close()
            if os.path.getsize(log_file) > 0:
                sup.create_csv_file(times, log_file, mode='a')
            else:
                sup.create_csv_file_header(times, log_file)

    def _define_response(self, settings, status, sim_values, **kwargs) -> None:
        response = dict()
        measurements = list()
        data = {'alg_manag': settings['alg_manag'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
        # Miner parms
        if settings['mining_alg'] == 'sm1':
            data['epsilon'] = settings['epsilon']
            data['eta'] = settings['eta']
        elif settings['mining_alg'] == 'sm2':
            data['concurrency'] = settings['concurrency']
        else:
            raise ValueError(settings['mining_alg'])
        similarity = 0
        # response['params'] = settings
        response['output'] = settings['output']
        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in sim_values])
            loss = (1 - similarity)
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
            measurements.append({**{'similarity': 0,
                                    'sim_metric': 'dl',
                                    'status': response['status']},
                                 **data})
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response

    def _split_timeline(self, size: float, one_ts: bool) -> None:
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
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        self.log_valdn = (valdn.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))

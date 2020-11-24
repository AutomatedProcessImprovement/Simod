# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:48:57 2020

@author: Manuel Camargo
"""
import os
import subprocess

import numpy as np
import pandas as pd
from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL


import readers.log_reader as lr
from support_modules.writers import xes_writer as xes
from support_modules.writers import xml_writer as xml
from support_modules.analyzers import sim_evaluator as sim


import opt_structure.structure_miner as sm
import opt_structure.structure_params_miner as spm

import utils.support as sup
from utils.support import timeit

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
                        response['status'] = STATUS_FAIL
                return response
            return safety_check

    def __init__(self, settings, args, log):
        """constructor"""
        self.space = self.define_search_space(settings, args)
        self.log = log
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
        space = {**{'concurrency': hp.uniform('concurrency',
                                          args['concurrency'][0],
                                          args['concurrency'][1]),
                    'alg_manag': hp.choice('alg_manag',
                                            args['alg_manag']),
                    'gate_management': hp.choice('gate_management',
                                                  args['gate_management']),
                    },
                  **settings}
        return space

    def execute_trials(self):

        parameters, resource_table = spm.StructureParametersMiner.mine_resources(
            self.settings, self.log)
        
        def exec_pipeline(trial_stg):
            # Vars initialization
            status = STATUS_OK
            exec_times = dict()
            data = pd.DataFrame(self.log.data)
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
                                          parameters,
                                          resource_table,
                                          data,
                                          status=status,
                                          log_time=exec_times)
            status = rsp['status']
            data = rsp['values'] if status == STATUS_OK else data
            # Simulate model
            rsp = self._simulate(trial_stg,
                                data,
                                status=status,
                                log_time=exec_times)
            status = rsp['status']
            sim_values = rsp['values'] if status == STATUS_OK else sim_values
            # Save times
            self._save_times(exec_times, trial_stg)
            # Optimizer results
            rsp = self._define_response(trial_stg, status, sim_values)
            print("-- End of trial --")
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.args['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        print('------ Final results ------')
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                       .sort_values('loss', ascending=bool))
            self.best_output = (results[results.status=='ok']
                                .head(1).iloc[0].output)
            self.best_parms = best
        except Exception as e:
            print(e)
            pass
        print(self.best_output)
        print(self.best_parms)

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
        structure_miner = sm.StructureMiner(settings, self.log)
        structure_miner.execute_pipeline()
        if structure_miner.is_safe:
            return [structure_miner.bpmn, structure_miner.process_graph]
        else:
            raise RuntimeError('Mining Structure error')

    @timeit(rec_name='EXTRACTING_PARAMS')
    @Decorators.safe_exec
    def _extract_parameters(self, settings, structure, parameters, resource_table, temp_log, **kwargs) -> None:
        bpmn, process_graph = structure
        # TODO: verificar porque self.log y no temp_log???
        p_extractor = spm.StructureParametersMiner(self.log,
                                                   bpmn,
                                                   process_graph,
                                                   resource_table,
                                                   settings)
        num_inst = len(temp_log.caseid.unique())
        # Get minimum date
        start_time = (temp_log
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

            temp_log.rename(columns={'user': 'resource'}, inplace=True)
            temp_log['source'] = 'log'
            temp_log['run_num'] = 0
            temp_log = temp_log.merge(
                p_extractor.resource_table[['resource', 'role']],
                on='resource',
                how='left')
            temp_log = temp_log[
                ~temp_log.task.isin(['Start', 'End'])]
            return temp_log
        else:
            raise RuntimeError('Parameters extraction error')

    @timeit(rec_name='SIM_EVAL')
    @Decorators.safe_exec
    def _simulate(self, settings, data, **kwargs) -> list:
        sim_values = list()
        for rep in range(settings['repetitions']):
            print("Experiment #" + str(rep + 1))
            self._execute_simulator(settings, rep)
            data = data.append(
                self._read_stats(settings, rep),
                ignore_index=True,
                sort=False)
            evaluator = sim.SimilarityEvaluator(
                data,
                settings,
                rep)
            evaluator.measure_distance('dl')
            sim_values.append(evaluator.similarity)
        return sim_values

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
                'concurrency': settings['concurrency'],
                'gate_management': settings['gate_management'],
                'output': settings['output']}
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

    @staticmethod
    def _read_stats(settings, rep):
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
        temp = pd.DataFrame(temp.data)
        temp.rename(columns={'user': 'resource'}, inplace=True)
        temp['role'] = temp['resource']
        temp['source'] = 'simulation'
        temp['run_num'] = rep + 1
        temp = temp[~temp.task.isin(['Start', 'End'])]
        return temp

    @staticmethod
    def _execute_simulator(settings, rep):
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



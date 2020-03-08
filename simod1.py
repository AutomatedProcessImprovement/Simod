# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
import subprocess
import types

import pandas as pd
import numpy as np

from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

from support_modules import support as sup
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
        self.bpmn = types.SimpleNamespace()
        self.process_graph = types.SimpleNamespace()

        self.sim_values = list()
        self.response = dict()

    def execute_pipeline(self, mode) -> None:
        if mode in ['optimizer', 'tasks_optimizer']:
            self.temp_path_redef()
        if self.status == STATUS_OK:
            self.read_inputs()
            self.evaluate_alignment()
            # TODO raise exception
        if self.status == STATUS_OK:
            self.extract_parameters()
            self.simulate()
            # TODO raise exception
            self.mannage_results()

    def temp_path_redef(self):
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

    def read_inputs(self):
        # Output folder creation
        if not os.path.exists(self.settings['output']):
            os.makedirs(self.settings['output'])
            os.makedirs(os.path.join(self.settings['output'], 'sim_data'))
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.settings['input'],
                                             self.settings['file']),
                                self.settings['read_options'])
        # Create customized event-log for the external tools
        xes.XesWriter(self.log, self.settings)
        # Execution steps
        self.mining_structure(self.settings)
        self.bpmn = br.BpmnReader(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'.bpmn'))
        self.process_graph = gph.create_process_structure(self.bpmn)

    def evaluate_alignment(self):
        # Evaluate alignment
        try:
            chk.evaluate_alignment(self.process_graph,
                                   self.log,
                                   self.settings)
        except Exception as e:
            print(e)
            self.status = STATUS_FAIL

    def extract_parameters(self):
        print("-- Mining Simulation Parameters --")
        p_extractor = par.ParameterMiner(self.log,
                                         self.bpmn,
                                         self.process_graph,
                                         self.settings)
        p_extractor.extract_parameters()
        self.process_stats = p_extractor.process_stats
        # print parameters in xml bimp format
        xml.print_parameters(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0]+'.bpmn'),
            os.path.join(self.settings['output'],
                         self.settings['file'].split('.')[0]+'.bpmn'),
            p_extractor.parameters)
        self.process_stats = pd.DataFrame.from_records(self.process_stats)

    def simulate(self):
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

    def mannage_results(self):
        self.response, measurements = self.define_response(self.status,
                                                           self.sim_values,
                                                           self.settings)

        if self.settings['exec_mode'] in ['optimizer', 'tasks_optimizer']:
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
        if settings['exec_mode'] == 'tasks_optimizer':
            if settings['pdef_method'] == 'apx':
                data = settings['tasks']
            else:
                data = dict()
                for task in settings['percentage'].keys():
                    data[task+'_p'] = settings['percentage'][task]
                    data[task] = (settings['percentage'][task] *
                                  settings['enabling_times'][task])
        else:
            data = {'alg_manag': settings['alg_manag'],
                    'epsilon': settings['epsilon'],
                    'eta': settings['eta'],
                    'rp_similarity': settings['rp_similarity'],
                    'gate_management': settings['gate_management'],
                    'output': settings['output']}
        if settings['exec_mode'] in ['optimizer', 'tasks_optimizer']:
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


# =============================================================================
# Hyperparameter-optimizer execution
# =============================================================================

# def hyper_execution(settings, args):
#     """Execute splitminer for bpmn structure mining."""
#     space = {**{'epsilon': hp.uniform('epsilon',
#                                       args['epsilon'][0],
#                                       args['epsilon'][1]),
#                 'eta': hp.uniform('eta',
#                                   args['eta'][0],
#                                   args['eta'][1]),
#                 'alg_manag': hp.choice('alg_manag',
#                                        ['replacement',
#                                         'repair',
#                                         'removal']),
#                 'rp_similarity': hp.uniform('rp_similarity',
#                                             args['rp_similarity'][0],
#                                             args['rp_similarity'][1]),
#                 'gate_management': hp.choice('gate_management',
#                                              args['gate_management'])},
#              **settings}
#     ## Trials object to track progress
#     bayes_trials = Trials()
#     ## Optimize
#     best = fmin(fn=pipe_line_execution, space=space, algo=tpe.suggest,
#                 max_evals=args['max_eval'], trials=bayes_trials, show_progressbar=False)

#     print(best)

# # =============================================================================
# # Hyperparameter-optimizer execution
# # =============================================================================


# def task_hyper_execution(settings, args):
#     """Execute splitminer for bpmn structure mining."""
#     # Define initial enablig_times
#     stats = mine_max_enabling(settings)
#     act_stats = calculate_activities_stats(stats)
#     act_stats = act_stats.to_dict('records')
#     # Exclusion of automatic tasks
#     automatic = list()
#     if settings['file'] == 'ConsultaDataMining201618.xes':
#         automatic = ['Notificacion estudiante cancelacion soli',
#                      'Traer informacion estudiante - banner',
#                      'Transferir Creditos',
#                      'Transferir creditos homologables']
#         third_party = ['Radicar Solicitud Homologacion']
#     elif settings['file'] == 'PurchasingExample.xes':
#         automatic = ['Approve Purchase Order for payment',
#                      "Authorize Supplier's Invoice payment",
#                      'Choose best option',
#                      'Release Purchase Order',
#                      'Send Invoice']
#         third_party = ['Settle Conditions With Supplier',
#                        'Settle Dispute With Supplier']
#     # Automatic tasks removal
#     act_stats = [x for x in act_stats if x['task'] not in automatic]
#     # hp.normal(label, mu, sigma)
#     # hp.uniform(label, min, max)
#     # Define search_space
#     space = dict()
#     if settings['pdef_method'] == 'apx':
#         space['tasks'] = dict()
#         for task in act_stats:
#             mean = task['mean'] * 0.4
#             # As min and max values I restrict the variation to
#             # a 50% less and in excess not the minimum and maximum
#             space['tasks'][task['task']] = hp.uniform(
#                 task['task'],
#                 mean-(mean*0.5),
#                 mean+(mean*0.5))
#     elif settings['pdef_method'] == 'apx_percentage':
#         space['percentage'] = dict()
#         settings['enabling_times'] = dict()
#         for task in act_stats:
#             settings['enabling_times'][task['task']] = task['mean']
#             # If percentage, just define a value btw 0 and 1
#             if task['task'] not in third_party:
#                 space['percentage'][task['task']] = hp.uniform(
#                     task['task'], 0, 1)
#             else:
#                 space['percentage'][task['task']] = hp.uniform(
#                     task['task'], 0.9, 1)

#     space = {**space, **settings}
#     # [print(k, v) for k, v in settings.items()]

#     # Trials object to track progress
#     bayes_trials = Trials()
#     # Optimize
#     best = fmin(fn=pipe_line_execution, space=space, algo=tpe.suggest,
#                 max_evals=args['max_eval'],
#                 trials=bayes_trials,
#                 show_progressbar=False)
#     print(best)


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
                             settings['file'].split('.')[0]+'_'+str(rep+1)+'.csv')]
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
                                           temp,
                                           settings,
                                           source='simulation',
                                           run_num=rep + 1)
        temp_stats = results_replayer.process_stats
        temp_stats['role'] = temp_stats['resource']
        return temp_stats

# # =============================================================================
# # Tasks optizer methods definition
# # =============================================================================


# def mine_max_enabling(settings):
#     # Output folder creation
#     if not os.path.exists(settings['output']):
#         os.makedirs(settings['output'])
#         os.makedirs(os.path.join(settings['output'], 'sim_data'))
#     # Event log reading
#     log = lr.LogReader(os.path.join(settings['input'], settings['file']),
#                        settings['read_options'])
#     # Create customized event-log for the external tools
#     xes.XesWriter(log, settings)
#     # Execution steps
#     mining_structure(settings)
#     bpmn = br.BpmnReader(os.path.join(settings['output'],
#                                       settings['file'].split('.')[0]+'.bpmn'))
#     process_graph = gph.create_process_structure(bpmn)

#     _, _, temp_stats = rpl.replay(process_graph, log, settings, source='apx')
#     return pd.DataFrame(temp_stats)


# def calculate_activities_stats(temp_stats):
#     activities_table = (temp_stats[['duration', 'task']]
#                         .groupby(['task'])
#                         .agg(['min', 'max', 'mean', 'std'])
#                         .reset_index())
#     activities_table.columns = activities_table.columns.droplevel(0)
#     activities_table = activities_table.rename(index=str, columns={'': 'task'})
#     return activities_table

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
import subprocess

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


# =============================================================================
# Single execution
# =============================================================================
def pipe_line_execution(settings):
    if settings['exec_mode'] in ['optimizer', 'tasks_optimizer']:
        # Paths redefinition
        settings['output'] = os.path.join('outputs', sup.folder_id())
        if settings['alg_manag'] == 'repairment':
            settings['aligninfo'] = os.path.join(
                                                 settings['output'],
                                                 'CaseTypeAlignmentResults.csv'
                                                 )
            settings['aligntype'] = os.path.join(settings['output'],
                                                 'AlignmentStatistics.csv')
    # Output folder creation
    if not os.path.exists(settings['output']):
        os.makedirs(settings['output'])
        os.makedirs(os.path.join(settings['output'], 'sim_data'))
    # Event log reading
    log = lr.LogReader(os.path.join(settings['input'], settings['file']),
                        settings['read_options'])
    # Create customized event-log for the external tools
    file_name = settings['file'].split('.')[0]
    xes.create_xes_file(log, os.path.join(settings['output'], file_name+'.xes'),
                        settings['read_options'])
    # Execution steps
    mining_structure(settings)
    bpmn = br.BpmnReader(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'))
    process_graph = gph.create_process_structure(bpmn)

    # Evaluate alignment
    chk.evaluate_alignment(process_graph, log, settings)

    print("-- Mining Simulation Parameters --")
    parameters, process_stats = par.extract_parameters(log, bpmn, process_graph, settings)
    xml.print_parameters(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                          os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                          parameters)
    status = STATUS_OK
    sim_values = list()
    process_stats = pd.DataFrame.from_records(process_stats)
    for rep in range(settings['repetitions']):
        print("Experiment #" + str(rep + 1))
        try:
            simulate(settings, rep)
            process_stats = process_stats.append(read_stats(settings, bpmn, rep),
                                                 ignore_index=True,
                                                 sort=False)
            evaluation = sim.SimilarityEvaluator(process_stats, settings,
                                                 rep, metric='dl_mae')
            sim_values.append(evaluation.similarity)
        except Exception as e:
            print(e)
            status = STATUS_FAIL
            break

    response, measurements = define_response(status, sim_values, settings)

    if settings['exec_mode'] == 'optimizer':
        if os.path.getsize(os.path.join('outputs', settings['temp_file'])) > 0:
            sup.create_csv_file(measurements, os.path.join('outputs', settings['temp_file']),mode='a')
        else:
            sup.create_csv_file_header(measurements, os.path.join('outputs', settings['temp_file']))
    else:
        print('------ Final results ------')
        [print(k, v, sep=': ') for k, v in response.items()]
    # response = {'loss': 0.5, 'status': STATUS_OK}
    return response

def define_response(status, sim_values, settings):
    response = dict()
    measurements = list()
    data = {
        'alg_manag': settings['alg_manag'], 'epsilon': settings['epsilon'],
        'eta': settings['eta'], 'output': settings['output']
        }
    if settings['exec_mode'] == 'optimizer':
        if status == STATUS_OK:
            loss = (1 - np.mean([x['act_norm'] for x in sim_values]))
            if loss < 0:
                response = {'loss': loss, 'params': settings, 'status': STATUS_FAIL}
                measurements.append({**{'similarity': 1 - loss, 'status': STATUS_FAIL}, **data})
            else:
                response = {'loss': loss, 'params': settings, 'status': status}
                measurements.append({**{'similarity': 1 - loss, 'status': status}, **data})
        else:
            response = {'params': settings, 'status': status}
            measurements.append({**{'similarity': 0, 'status': status}, **data})
    else:
        if status == STATUS_OK:
            similarity = (np.mean([x['act_norm'] for x in sim_values]))
            if similarity < 0:
                response = {**{'similarity': similarity, 'status': STATUS_FAIL}, **data}
            else:
                response = {**{'similarity': similarity, 'status': status}, **data}
        else:
            response = {**{'similarity': 0, 'status': status}, **data}
    return response, measurements


# =============================================================================
# Hyperparameter-optimizer execution
# =============================================================================

def hyper_execution(settings, args):
    """Execute splitminer for bpmn structure mining."""
    space = {**{'epsilon': hp.uniform('epsilon', args['epsilon'][0], args['epsilon'][1]),
             'eta': hp.uniform('eta', args['eta'][0], args['eta'][1]),
             'alg_manag': hp.choice('alg_manag', ['replacement',
                                                  'repairment',
                                                  'removal'])}, **settings}
    ## Trials object to track progress
    bayes_trials = Trials()
    ## Optimize
    best = fmin(fn=pipe_line_execution, space=space, algo=tpe.suggest,
                max_evals=args['max_eval'], trials=bayes_trials, show_progressbar=False)

    print(best)

# =============================================================================
# Hyperparameter-optimizer execution
# =============================================================================

def task_hyper_execution(settings, args):
    """Execute splitminer for bpmn structure mining."""
    # TODO: define initial enablig_times
    stats = mine_max_enabling(settings)
    act_stats = calculate_activities_stats(stats)
    act_stats = act_stats.to_dict('records')
    # TODO: define search_space
    # hp.normal(label, mu, sigma)
    # hp.uniform(label, min, max)
    space = dict()
    space['tasks'] = dict()
    for task in act_stats:
        space['tasks'][task['task']] = hp.uniform(task['task'], task['min'], task['max'])
    space = {**space, **settings}
    # [print(k, v) for k, v in settings.items()]

    # TODO: define evaluation metric maybe in relation with cycletime

    ## Trials object to track progress
    bayes_trials = Trials()
    ## Optimize
    best = fmin(fn=pipe_line_execution, space=space, algo=tpe.suggest,
                max_evals=args['max_eval'], trials=bayes_trials, show_progressbar=False)
    print(best)


# =============================================================================
# External tools calling
# =============================================================================

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

def simulate(settings, rep):
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

def read_stats(settings, bpmn, rep):
    """Reads the simulation results stats
    Args:
        settings (dict): Path to jar and file names
        rep (int): repetition number
    """
    m_settings = dict()
    m_settings['output'] = settings['output']
    m_settings['file'] = settings['file']
    column_names = {'resource':'user'}
    m_settings['read_options'] = settings['read_options']
    m_settings['read_options']['timeformat'] = '%Y-%m-%d %H:%M:%S.%f'
    m_settings['read_options']['column_names'] = column_names
    temp = lr.LogReader(os.path.join(m_settings['output'], 'sim_data',
                                     m_settings['file'].split('.')[0] + '_'+str(rep + 1)+'.csv'), m_settings['read_options'])
    process_graph = gph.create_process_structure(bpmn)
    _, _, temp_stats = rpl.replay(process_graph, temp, settings, source='simulation', run_num=rep + 1)
    temp_stats = pd.DataFrame.from_records(temp_stats)
    temp_stats['role'] = temp_stats['resource']
    return temp_stats

# =============================================================================
# Tasks optizer methods definition
# =============================================================================
def mine_max_enabling(settings):
    # Output folder creation
    if not os.path.exists(settings['output']):
        os.makedirs(settings['output'])
        os.makedirs(os.path.join(settings['output'], 'sim_data'))
    # Event log reading
    log = lr.LogReader(os.path.join(settings['input'], settings['file']),
                        settings['read_options'])
    # Create customized event-log for the external tools
    file_name = settings['file'].split('.')[0]
    xes.create_xes_file(log, os.path.join(settings['output'], file_name+'.xes'),
                        settings['read_options'])
    # Execution steps
    mining_structure(settings)
    bpmn = br.BpmnReader(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'))
    process_graph = gph.create_process_structure(bpmn)

    _, _, temp_stats = rpl.replay(process_graph, log, settings, source='apx')
    # if os.path.exists(settings['output']):
    #     os.remove(settings['output'])
    return pd.DataFrame(temp_stats)

def calculate_activities_stats(temp_stats):
    activities_table = temp_stats[['duration','task']].groupby(['task']).agg(['min','max','mean','std']).reset_index()
    activities_table.columns = activities_table.columns.droplevel(0)
    activities_table = activities_table.rename(index=str, columns={'': 'task'})
    return activities_table


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
#import sys
import re
import os
import subprocess
import configparser as cp
from shutil import copyfile

import pandas as pd
import numpy as np

from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

from support_modules import support as sup
from support_modules.readers import log_reader as lr
from support_modules.readers import bpmn_reader as br
from support_modules.readers import process_structure as gph
from support_modules.writers import xml_writer as xml
from support_modules.analyzers import generalization as gen
from support_modules.log_repairing import conformance_checking as chk

from extraction import parameter_extraction as par
from extraction import log_replayer as rpl


# =============================================================================
# Single execution
# =============================================================================

def single_exec(settings):
    """Main aplication method"""
    # Read settings from config file
    settings = read_settings(settings)
    # Output folder creation
    if not os.path.exists(settings['output']):
        os.makedirs(settings['output'])
        os.makedirs(os.path.join(settings['output'], 'sim_data'))
    # Copy event-log to output folder
    copyfile(os.path.join(settings['input'], settings['file']),
             os.path.join(settings['output'], settings['file']))
    # Event log reading
    log = lr.LogReader(os.path.join(settings['output'], settings['file']),
                       settings['timeformat'])
    # Execution steps
    mining_structure(settings, settings['epsilon'], settings['eta'])
    bpmn = br.BpmnReader(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'))
    process_graph = gph.create_process_structure(bpmn)

    # Evaluate alignment
    chk.evaluate_alignment(process_graph, log, settings)

    print("-- Mining Simulation Parameters --")
    parameters, process_stats = par.extract_parameters(log, bpmn, process_graph)
    xml.print_parameters(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                         os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                         parameters)
    response = list()
    status = 'ok'
    sim_values = list()
    if settings['simulation']:
#        if settings['analysis']:
        process_stats = pd.DataFrame.from_records(process_stats)
        for rep in range(settings['repetitions']):
            print("Experiment #" + str(rep + 1))
            try:
                simulate(settings, rep)
                process_stats = process_stats.append(measure_stats(settings,
                                                                   bpmn, rep),
                                                     ignore_index=True,
                                                     sort=False)
                sim_values.append(gen.mesurement(process_stats, settings, rep))
            except:
                status = 'fail'
                break
    data = {'alg_manag': settings['alg_manag'],
            'epsilon': settings['epsilon'],
            'eta': settings['eta'],
            'output': settings['output']
            }

    if status == 'ok':
        loss = (1 - np.mean([x['act_norm'] for x in sim_values]))
        if loss < 0:
            response.append({**{'loss': loss, 'status': 'fail'}, **data})
        else:
            response.append({**{'loss': loss, 'status': status}, **data})
    else:
        response.append({**{'loss': 1, 'status': status}, **data})

    return response

# =============================================================================
# Hyper-optimaizer execution
# =============================================================================

def objective(settings):
    """Main aplication method"""
    # Read settings from config file
    settings = read_settings(settings)
    # Output folder creation
    if not os.path.exists(settings['output']):
        os.makedirs(settings['output'])
        os.makedirs(os.path.join(settings['output'], 'sim_data'))
    # Copy event-log to output folder
    copyfile(os.path.join(settings['input'], settings['file']),
             os.path.join(settings['output'], settings['file']))
    # Event log reading
    log = lr.LogReader(os.path.join(settings['output'], settings['file']),
                       settings['timeformat'])
    # Execution steps
    mining_structure(settings, settings['epsilon'], settings['eta'])
    bpmn = br.BpmnReader(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'))
    process_graph = gph.create_process_structure(bpmn)

    # Evaluate alignment
    chk.evaluate_alignment(process_graph, log, settings)

    print("-- Mining Simulation Parameters --")
    parameters, process_stats = par.extract_parameters(log, bpmn, process_graph)
    xml.print_parameters(os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                         os.path.join(settings['output'],
                                      settings['file'].split('.')[0]+'.bpmn'),
                         parameters)
    response = dict()
    measurements = list()
    status = STATUS_OK
    sim_values = list()
    process_stats = pd.DataFrame.from_records(process_stats)
    for rep in range(settings['repetitions']):
        print("Experiment #" + str(rep + 1))
        try:
            simulate(settings, rep)
            process_stats = process_stats.append(measure_stats(settings,
                                                               bpmn, rep),
                                                 ignore_index=True,
                                                 sort=False)
            sim_values.append(gen.mesurement(process_stats, settings, rep))
        except:
            status = STATUS_FAIL
            break

    data = {'alg_manag': settings['alg_manag'],
            'epsilon': settings['epsilon'],
            'eta': settings['eta'],
            'output': settings['output']
            }
    if status == STATUS_OK:
        loss = (1 - np.mean([x['act_norm'] for x in sim_values]))
        if loss < 0:
            response = {'loss': loss, 'params': settings, 'status': STATUS_FAIL}
            measurements.append({**{'loss': loss, 'status': STATUS_FAIL}, **data})
        else:
            response = {'loss': loss, 'params': settings, 'status': status}
            measurements.append({**{'loss': loss, 'status': status}, **data})
    else:
        response = {'params': settings, 'status': status}
        measurements.append({**{'loss': 1, 'status': status}, **data})
   
    if os.path.getsize(os.path.join('outputs', settings['temp_file'])) > 0:
        sup.create_csv_file(measurements, os.path.join('outputs', settings['temp_file']),mode='a')
    else:
        sup.create_csv_file_header(measurements, os.path.join('outputs', settings['temp_file']))
    return response


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
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=args['max_eval'], trials=bayes_trials, show_progressbar=False)
    
    measurements = list()
    for res in bayes_trials.results:
        measurements.append({
            'loss': res['loss'],
            'alg_manag': res['params']['alg_manag'],
            'epsilon': res['params']['epsilon'],
            'eta': res['params']['eta'],
            'status': res['status'],
            'output': res['params']['output']
            })
    return best, measurements
        # Save results
#        measurements = list()
#        for res in bayes_trials.results:
#            measurements.append({
#                'loss': res['loss'],
#                'alg_manag': res['params']['alg_manag'],
#                'epsilon': res['params']['epsilon'],
#                'eta': res['params']['eta'],
#                'status': res['status'],
#                'output': res['params']['output']
#                })
#        config = cp.ConfigParser(interpolation=None)
#        config.read("./config.ini")
#        sup.create_csv_file_header(measurements,
#                                   os.path.join(config.get('FOLDERS', 'outputs'),
#                                                config.get('EXECUTION', 'filename')
#                                                .split('.')[0]+'_'+
#                                                sup.folder_id()+'.csv'))

# =============================================================================
# External tools calling
# =============================================================================

def mining_structure(settings, epsilon, eta):
    """Execute splitminer for bpmn structure mining.
    Args:
        settings (dict): Path to jar and file names
        epsilon (double): Parallelism threshold (epsilon) in [0,1]
        eta (double): Percentile for frequency threshold (eta) in [0,1]
    """
    print(" -- Mining Process Structure --")
    args = ['java', '-jar', settings['miner_path'],
            str(epsilon), str(eta),
            os.path.join(settings['output'], settings['file']),
            os.path.join(settings['output'], settings['file'].split('.')[0])]
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

def measure_stats(settings, bpmn, rep):
    """Executes BIMP Simulations.
    Args:
        settings (dict): Path to jar and file names
        rep (int): repetition number
    """
    timeformat = '%Y-%m-%d %H:%M:%S.%f'
    temp = lr.LogReader(os.path.join(settings['output'], 'sim_data',
                                     settings['file'].split('.')[0] + '_'+str(rep + 1)+'.csv'),
                        timeformat)
    process_graph = gph.create_process_structure(bpmn)
    _, _, temp_stats = rpl.replay(process_graph, temp, source='simulation', run_num=rep + 1)
    temp_stats = pd.DataFrame.from_records(temp_stats)
    role = lambda x: x['resource']
    temp_stats['role'] = temp_stats.apply(role, axis=1)
    return temp_stats

# =============================================================================
# Support
# =============================================================================

def reformat_path(raw_path):
    """Provides path support to different OS path definition"""
    route = re.split(chr(92)+'|'+chr(92)+chr(92)+'|'+
                     chr(47)+'|'+chr(47)+chr(47), raw_path)
    return os.path.join(*route)

def read_settings(settings):
    """Catch parameters fron console or code defined"""
    config = cp.ConfigParser(interpolation=None)
    config.read("./config.ini")
    # Basic settings
    settings['input'] = config.get('FOLDERS', 'inputs')
    settings['output'] = os.path.join(config.get('FOLDERS', 'outputs'), sup.folder_id())
    settings['timeformat'] = config.get('EXECUTION', 'timeformat')
    # Conditional settings
    settings['miner_path'] = reformat_path(config.get('EXTERNAL', 'splitminer'))
    if settings['alg_manag'] == 'repairment':
        settings['align_path'] = reformat_path(config.get('EXTERNAL', 'proconformance'))
        settings['aligninfo'] = os.path.join(settings['output'],
                                             config.get('ALIGNMENT', 'aligninfo'))
        settings['aligntype'] = os.path.join(settings['output'],
                                             config.get('ALIGNMENT', 'aligntype'))
    if settings['simulation']:
        settings['bimp_path'] = reformat_path(config.get('EXTERNAL', 'bimp'))
    return settings
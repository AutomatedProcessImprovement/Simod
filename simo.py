# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import sys
import re
import os
import subprocess
import configparser as cp
import getopt
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

def objective(params):
    """Main aplication method"""
    # Read settings from config file
    settings = read_settings(params)
    params['output'] = settings['output']
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
    if settings['mining']:
        mining_structure(settings, params['epsilon'], params['eta'])
        bpmn = br.BpmnReader(os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'))
        process_graph = gph.create_process_structure(bpmn)

    if settings['alignment']:
        # Evaluate alignment
        chk.evaluate_alignment(process_graph, log, settings)

    if settings['parameters']:
        print("-- Mining Simulation Parameters --")
        parameters, process_stats = par.extract_parameters(log, bpmn, process_graph)
        xml.print_parameters(os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'),
                             os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'),
                             parameters)
    response = dict()
    status = STATUS_OK
    sim_values = list()
    if settings['simulation']:
        if settings['analysis']:
            process_stats = pd.DataFrame.from_records(process_stats)
        for rep in range(settings['repetitions']):
            print("Experiment #" + str(rep + 1))
            try:
                simulate(settings, rep)
                if settings['analysis']:
                    process_stats = process_stats.append(measure_stats(settings,
                                                                       bpmn, rep),
                                                         ignore_index=True,
                                                         sort=False)
                    sim_values.append(gen.mesurement(process_stats, settings, rep))
            except:
                status = STATUS_FAIL
                break

    if status == STATUS_OK:
        loss = (1 - np.mean([x['act_norm'] for x in sim_values]))
        if loss < 0:
            response = {'loss': loss, 'params': params, 'status': STATUS_FAIL}
        else:
            response = {'loss': loss, 'params': params, 'status': status}
    else:
        response = {'params': params, 'status': status}
    return response


def main(argv):
    """Execute splitminer for bpmn structure mining."""
    if argv:
        parameters = define_parameter(argv)
        response = objective(parameters)
        print('Results:')
        print(response)
    else:
        space = {'epsilon': hp.uniform('epsilon', 0.0, 1.0),
                 'eta': hp.uniform('eta', 0.0, 1.0),
                 'alg_manag': hp.choice('alg_manag', ['replacement',
                                                      'trace_alignment',
                                                      'removal'])}
        ## Trials object to track progress
        bayes_trials = Trials()
        max_evals = 50
        ## Optimize
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=bayes_trials, show_progressbar=False)
    
        print(best)
        # Save results
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
        config = cp.ConfigParser(interpolation=None)
        config.read("./config.ini")
        sup.create_csv_file_header(measurements,
                                   os.path.join(config.get('FOLDERS', 'outputs'),
                                                config.get('EXECUTION', 'filename')
                                                .split('.')[0]+'_'+
                                                sup.folder_id()+'.csv'))

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

def read_settings(params):
    """Catch parameters fron console or code defined"""
    settings = dict()
    valid_opt = ['true', 'True', '1', 'Yes', 'yes']
    config = cp.ConfigParser(interpolation=None)
    config.read("./config.ini")
    # Basic settings
    settings['input'] = config.get('FOLDERS', 'inputs')
    settings['output'] = os.path.join(config.get('FOLDERS', 'outputs'), sup.folder_id())
    settings['file'] = config.get('EXECUTION', 'filename')
    settings['timeformat'] = config.get('EXECUTION', 'timeformat')
    settings['mining'] = config.get('EXECUTION', 'mining') in valid_opt
    settings['alignment'] = config.get('EXECUTION', 'alignment') in valid_opt
    settings['parameters'] = config.get('EXECUTION', 'parameters') in valid_opt
    settings['simulation'] = config.get('EXECUTION', 'simulation') in valid_opt
    settings['analysis'] = config.get('EXECUTION', 'analysis') in valid_opt
    settings['repetitions'] = int(config.get('EXECUTION', 'repetitions'))
    # Conditional settings
    if settings['mining']:
        settings['miner_path'] = reformat_path(config.get('EXTERNAL', 'splitminer'))
    if settings['alignment']:
        settings['alg_manag'] = params['alg_manag']
        if settings['alg_manag'] == 'trace_alignment':
            settings['align_path'] = reformat_path(config.get('EXTERNAL', 'proconformance'))
            settings['aligninfo'] = os.path.join(settings['output'],
                                                 config.get('ALIGNMENT', 'aligninfo'))
            settings['aligntype'] = os.path.join(settings['output'],
                                                 config.get('ALIGNMENT', 'aligntype'))
    if settings['simulation']:
        settings['bimp_path'] = reformat_path(config.get('EXTERNAL', 'bimp'))
        if settings['analysis']:
            settings['repetitions'] = int(config.get('EXECUTION', 'repetitions'))
    return settings


def define_parameter(argv):
    """Catch parameters fron console or code defined"""
    parameters = dict()
    switch = {'-e':'epsilon', '-n':'eta', '-m':'alg_manag'}
    try:
        opts, _ = getopt.getopt(argv, "he:n:m:",
                                ["epsilon=", "eta=", "alg_manag="])
        for opt, arg in opts:
            parameters[switch[opt]] = arg
    except getopt.GetoptError:
        print('Invalid option')
        sys.exit(2)
    return parameters

# =============================================================================
# Main function
# =============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])

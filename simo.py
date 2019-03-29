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
from shutil import copyfile
from support_modules import support as sup
from support_modules.readers import readers as rd
from support_modules.readers import log_reader as lr
from support_modules.readers import bpmn_reader as br
from support_modules.writers import xml_writer as xml

from support_modules.log_repairing import traces_alignment as tal
from extraction import parameter_extraction as par



def main(argv):
    """Main aplication method"""
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

    # Output folder creation
    if not os.path.exists(settings['output']):
        os.makedirs(settings['output'])
        os.makedirs(os.path.join(settings['output'], 'sim_data'))
    # Copy event-log to output folder
    copyfile(os.path.join(settings['input'], settings['file']),
             os.path.join(settings['output'], settings['file']))
    # Event log reading
    log = lr.LogReader(os.path.join(settings['output'], settings['file']),
                       settings['timeformat'], settings['timeformat'])
    # Execution steps
    if settings['mining']:
        settings['miner_path'] = reformat_path(config.get('EXTERNAL', 'splitminer'))
        # TODO: variar parametros para experimentacion
        mining_structure(settings, '0.1', '0.8')
        bpmn = br.BpmnReader(os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'))
    if settings['alignment']:
        settings['align_path'] = reformat_path(config.get('EXTERNAL', 'proconformance'))
        # Evaluate alignment
        evaluate_alignment(settings)
        # Repare event-log
        tal.align_traces(log,
                         os.path.join(settings['output'], config.get('ALIGNMENT', 'aligninfo')),
                         os.path.join(settings['output'], config.get('ALIGNMENT', 'aligntype')))
    if settings['parameters']:
        print("-- Mining Simulation Parameters --")
        parameters, process_stats = par.extract_parameters(log, bpmn)
        xml.print_parameters(os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'),
                             os.path.join(settings['output'],
                                          settings['file'].split('.')[0]+'.bpmn'),
                             parameters)
    if settings['simulation']:
        settings['bimp_path'] = reformat_path(config.get('EXTERNAL', 'bimp'))
        for rep in range(int(config.get('EXECUTION', 'simcycles'))):
            print("Experiment #" + str(rep + 1))
            simulate(settings, rep)
        bimp_statistics = rd.import_bimp_statistics(os.path.join(settings['output'], 'sim_data'),
                                                    os.path.join(settings['output'],
                                                                 settings['file'].split('.')[0]+'.bpmn'))
    print(bimp_statistics)

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
            epsilon, eta,
            os.path.join(settings['output'], settings['file']),
            os.path.join(settings['output'], settings['file'].split('.')[0])]
    subprocess.call(args)

def evaluate_alignment(settings):
    """Evaluate business process traces alignment in relation with BPMN structure.
    Args:
        settings (dict): Path to jar and file names
    """
    print(" -- Evaluating event log alignment --")
    args = ['java', '-jar', settings['align_path'],
            settings['output']+os.sep,
            settings['file'],
            settings['file'].split('.')[0]+'.bpmn',
            'true']
    subprocess.call(args, bufsize=-1)

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

# =============================================================================
# Support
# =============================================================================

def reformat_path(raw_path):
    """Provides path support to different OS path definition"""
    route = re.split(chr(92)+'|'+chr(92)+chr(92)+'|'+
                     chr(47)+'|'+chr(47)+chr(47), raw_path)
    return os.path.join(*route)

if __name__ == "__main__":
    main(sys.argv[1:])

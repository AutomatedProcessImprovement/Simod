# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:25:10 2019

@author: Manuel Camargo
"""
import os
import sys
import getopt
import simod as sim

from support_modules import support as sup

# =============================================================================
# Main function
# =============================================================================


def main(argv):
    settings = dict()
    args = dict()
    # Similarity btw the resources profile execution (Song e.t. all)
    settings['rp_similarity'] = 0.5
    settings = define_general_settings(settings)
    # Exec mode 'single', 'optimizer', 'tasks_optimizer'
    settings['exec_mode'] = 'single'
    # Similarity metric 'tsd', 'dl_mae', 'tsd_min'
    settings['sim_metric'] = 'tsd_min'
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        settings['file'] = 'PurchasingExample.xes.gz'
        settings['repetitions'] = 1
        settings['simulation'] = True
        if settings['exec_mode'] == 'single':
            # Splitminer settings [0..1]
            settings['epsilon'] = 0.97
            settings['eta'] = 0.55
            # 'removal', 'replacement', 'repairment'
            settings['alg_manag'] = 'removal'
            # Processing time definition method:
            # 'manual', 'automatic', 'semi-automatic'
            settings['pdef_method'] = 'semi-automatic'
            # Single Execution
            # sim.single_exec(settings)
            sim.pipe_line_execution(settings)
        elif settings['exec_mode'] == 'optimizer':
            args['epsilon'] = [0.0, 1.0]
            args['eta'] = [0.0, 1.0]
            args['max_eval'] = 50
            settings['temp_file'] = sup.file_id(prefix='OP_')
            settings['pdef_method'] = 'automatic'
            # Execute optimizer
            if not os.path.exists(os.path.join('outputs',
                                               settings['temp_file'])):
                open(os.path.join('outputs',
                                  settings['temp_file']), 'w').close()
                sim.hyper_execution(settings, args)
        elif settings['exec_mode'] == 'tasks_optimizer':
            # Splitminer settings [0..1]
            settings['epsilon'] = 0.97
            settings['eta'] = 0.55
            # 'removal', 'replacement', 'repairment'
            settings['alg_manag'] = 'removal'
            # Processing time definition method: 'apx'
            settings['pdef_method'] = 'apx'
            args['max_eval'] = 50
            settings['temp_file'] = sup.file_id(prefix='TS_')
            # Execute optimizer
            if not os.path.exists(os.path.join('outputs',
                                               settings['temp_file'])):
                open(os.path.join('outputs',
                                  settings['temp_file']), 'w').close()
                sim.task_hyper_execution(settings, args)
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(argv, "hf:e:n:m:r:",
                                    ['eventlog=', "epsilon=", "eta=",
                                     "alg_manag=", "repetitions="])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['epsilon', 'eta']:
                    settings[key] = float(arg)
                elif key == 'repetitions':
                    settings[key] = int(arg)
                else:
                    settings[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
        settings['simulation'] = True
        sim.single_exec(settings)
# =============================================================================
# Support
# =============================================================================


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file', '-e': 'epsilon',
              '-n': 'eta', '-m': 'alg_manag', '-r': 'repetitions'}
    try:
        return switch[opt]
    except Exception as e:
        print(e.message)
        raise Exception('Invalid option ' + opt)


def define_general_settings(settings):
    """ Sets the app general settings"""
    column_names = {'Case ID': 'caseid', 'Activity': 'task',
                    'lifecycle:transition': 'event_type', 'Resource': 'user'}
    # Event-log reading options
    settings['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                'column_names': column_names,
                                'one_timestamp': False,
                                'reorder': False,
                                'filter_d_attrib': True,
                                'ns_include': True}
    # Folders structure
    settings['input'] = 'inputs'
    settings['output'] = os.path.join('outputs', sup.folder_id())
    # External tools routes
    settings['miner_path'] = os.path.join('external_tools',
                                          'splitminer',
                                          'splitminer.jar')
    settings['bimp_path'] = os.path.join('external_tools',
                                         'bimp',
                                         'qbp-simulator-engine.jar')
    settings['align_path'] = os.path.join('external_tools',
                                          'proconformance',
                                          'ProConformance2.jar')
    settings['aligninfo'] = os.path.join(settings['output'],
                                         'CaseTypeAlignmentResults.csv')
    settings['aligntype'] = os.path.join(settings['output'],
                                         'AlignmentStatistics.csv')
    return settings


if __name__ == "__main__":
    main(sys.argv[1:])

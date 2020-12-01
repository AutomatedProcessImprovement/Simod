# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:25:10 2019

@author: Manuel Camargo
"""
import os
import sys
import getopt
import simod as sim

import utils.support as sup

# =============================================================================
# Main function
# =============================================================================


def main(argv):
    settings = dict()
    args = dict()
    settings = define_general_settings(settings)
    # Exec mode 'single', 'optimizer'
    settings['exec_mode'] = 'optimizer'
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        settings['file'] = 'poc_processmining.xes'
        settings['repetitions'] = 5
        settings['simulation'] = True
        if settings['exec_mode'] == 'single':
            # Similarity metric 'tsd', 'dl_mae', 'tsd_min', 'mae',
            # 'hour_emd', 'day_emd', 'day_hour_emd', 'cal_emd'
            settings['sim_metric'] = 'tsd' # Main metric
            # Additional metrics
            # settings['add_metrics'] = ['day_hour_emd', 'hour_emd', 'day_emd',
            #                             'cal_emd', 'log_mae', 'dl_mae', 'mae']
            settings['gate_management'] = 'discovery'
            # Similarity btw the resources profile execution (Song e.t. all)
            settings['rp_similarity'] = 0.672644226
            # Splitminer settings [0..1]
            # settings['concurrency'] = 0.5
            # Splitminer settings [0..1] default epsilon = 0.1, eta = 0.4
            settings['epsilon'] = 0.601063585
            settings['eta'] = 0.707803144
            # 'removal', 'replacement', 'repair'
            settings['alg_manag'] = 'removal'
            # Processing time definition method:
            # 'manual', 'automatic', 'semi-automatic'
            settings['pdef_method'] = 'automatic'
            # Calendar parameters
            # calendar methods 'default', 'discovered' ,'pool'
            settings['res_cal_met'] = 'default'
            if settings['res_cal_met'] == 'default':
                settings['res_dtype'] = '247'  # 'LV917', '247'
            else:
                settings['res_support'] = 0.1  # [0..1]
                settings['res_confidence'] = 50  # [50..85]
            # calendar methods 'default', 'discovered'
            settings['arr_cal_met'] = 'discovered'
            if settings['arr_cal_met'] == 'default':
                settings['arr_dtype'] = '247'  # 'LV917', '247'
            else:
                settings['arr_support'] = 0.1  # [0..1]
                settings['arr_confidence'] = 10  # [50..85]
            # temporal file for results
            settings['temp_file'] = sup.file_id(prefix='SE_')
            # Single Execution
            simod = sim.Simod(settings)
            simod.execute_pipeline()
        elif settings['exec_mode'] == 'optimizer':
            settings['sim_metric'] = 'tsd'
            settings['add_metrics'] = ['day_hour_emd', 'hour_emd', 'day_emd',
                                       'cal_emd', 'log_mae', 'dl_mae', 'mae']
            args['max_eval'] = 30
            # args['concurrency'] = [0.0, 1.0]
            args['epsilon'] = [0.0, 1.0]
            args['eta'] = [0.0, 1.0]
            args['alg_manag'] = ['replacement', 'repair', 'removal']
            args['rp_similarity'] = [0.5, 0.9]
            args['gate_management'] = ['discovery', 'equiprobable']
            # settings['gate_management'] = 'discovery'
            args['res_dtype'] = ['LV917', '247']
            args['arr_dtype'] = ['LV917', '247']
            args['res_sup_dis'] = [0.01, 0.3]  # [0..1]
            args['res_con_dis'] = [50, 85]  # [50..85]
            args['arr_support'] = [0.01, 0.1]  # [0..1]
            args['arr_confidence'] = [1, 20]  # [50..85]
            optimizer = sim.DiscoveryOptimizer(settings, args)
            optimizer.execute_pipeline()
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
                                'filter_d_attrib': True,
                                'ns_include': True}
    # Folders structure
    settings['input'] = 'inputs'
    settings['output'] = os.path.join('outputs', sup.folder_id())
    # External tools routes
    # settings['miner_path'] = os.path.join('external_tools',
    #                                       'splitminer',
    #                                       'sm2.jar')
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
    settings['calender_path'] = os.path.join('external_tools',
                                             'calenderimp',
                                             'CalenderImp.jar')
    settings['simulator'] = 'bimp'
    return settings


if __name__ == "__main__":
    main(sys.argv[1:])

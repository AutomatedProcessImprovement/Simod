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
    settings['gl']['exec_mode'] = 'optimizer'
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        settings['gl']['file'] = 'PurchasingExample.xes'
        settings['gl']['repetitions'] = 5
        settings['gl']['simulation'] = True
        if settings['gl']['exec_mode'] == 'single':
            # Similarity metric 'tsd', 'dl_mae', 'tsd_min', 'mae',
            # 'hour_emd', 'day_emd', 'day_hour_emd', 'cal_emd'
            settings['sim_metric'] = 'tsd' # Main metric
            # Additional metrics
            # settings['add_metrics'] = ['day_hour_emd', 'hour_emd', 'day_emd',
            #                             'cal_emd', 'log_mae', 'dl_mae', 'mae']
            settings['gate_management'] = 'equiprobable'
            # Similarity btw the resources profile execution (Song e.t. all)
            settings['rp_similarity'] = 0.672644226
            # Splitminer settings [0..1]
            settings['concurrency'] = 0.5
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
        elif settings['gl']['exec_mode'] == 'optimizer':
            settings['gl']['sim_metric'] = 'tsd'
            settings['gl']['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
            settings['strc'] = dict()
            settings['strc']['max_eval_s'] = 1
            settings['strc']['concurrency'] = [0.0, 1.0]
            settings['strc']['epsilon'] = [0.0, 1.0]
            settings['strc']['eta'] = [0.0, 1.0]
            settings['strc']['alg_manag'] = ['replacement', 'repair', 'removal']
            settings['strc']['gate_management'] = ['discovery', 'equiprobable']
            settings['tm'] = dict()
            settings['tm']['max_eval_t'] = 2
            settings['tm']['rp_similarity'] = [0.5, 0.9]
            settings['tm']['res_dtype'] = ['LV917', '247']
            settings['tm']['arr_dtype'] = ['LV917', '247']
            settings['tm']['res_sup_dis'] = [0.01, 0.3]  # [0..1]
            settings['tm']['res_con_dis'] = [50, 85]  # [50..85]
            settings['tm']['arr_support'] = [0.01, 0.1]  # [0..1]
            settings['tm']['arr_confidence'] = [1, 20]  # [50..85]
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
    settings['gl'] = dict()
    column_names = {'Case ID': 'caseid', 'Activity': 'task',
                    'lifecycle:transition': 'event_type', 'Resource': 'user'}
    # Event-log reading options
    settings['gl']['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                      'column_names': column_names,
                                      'one_timestamp': False,
                                      'filter_d_attrib': True}
    # Folders structure
    settings['gl']['input'] = 'inputs'
    settings['gl']['output'] = os.path.join('outputs', sup.folder_id())
    # External tools routes
    settings['gl']['sm2_path'] = os.path.join('external_tools',
                                              'splitminer2',
                                              'sm2.jar')
    settings['gl']['sm1_path'] = os.path.join('external_tools',
                                              'splitminer',
                                              'splitminer.jar')
    settings['gl']['sm3_path'] = os.path.join('external_tools',
                                              'splitminer3',
                                              'bpmtk.jar')
    settings['gl']['bimp_path'] = os.path.join('external_tools',
                                               'bimp',
                                               'qbp-simulator-engine.jar')
    settings['gl']['align_path'] = os.path.join('external_tools',
                                                'proconformance',
                                                'ProConformance2.jar')
    settings['gl']['aligninfo'] = os.path.join(settings['gl']['output'],
                                               'CaseTypeAlignmentResults.csv')
    settings['gl']['aligntype'] = os.path.join(settings['gl']['output'],
                                               'AlignmentStatistics.csv')
    settings['gl']['calender_path'] = os.path.join('external_tools',
                                                   'calenderimp',
                                                   'CalenderImp.jar')
    settings['gl']['simulator'] = 'bimp'
    settings['gl']['mining_alg'] = 'sm3'
    return settings


if __name__ == "__main__":
    main(sys.argv[1:])

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
    settings = define_general_settings(settings)
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        settings['file'] = 'PurchasingExample.xes'
        settings['mining_alg'] = 'sm3'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt( argv, "h:f:m:", ['file=', 'mining_alg='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                settings[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    settings['repetitions'] = 1
    settings['simulation'] = True
    settings['sim_metric'] = 'tsd'
    settings['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    settings['concurrency'] = 0.0
    settings['epsilon'] = 0.5
    settings['eta'] = 0.7
    # 'replacement', 'repair', 'removal'
    settings['alg_manag'] = 'replacement'
    # 'discovery', 'equiprobable'
    settings['gate_management'] = 'discovery'
    settings['rp_similarity'] = 0.8
    # 'discovered', 'default'
    settings['res_cal_met'] = 'discovered'
    settings['arr_cal_met'] = 'discovered'
    # 'LV917', '247'
    settings['res_dtype'] = '247'
    settings['arr_dtype'] = '247'
    # 'manual', 'automatic', 'semi-automatic'
    settings['pdef_method'] = 'automatic'
    settings['res_support'] = 0.05  # [0..1]
    settings['res_confidence'] = 70    # [50..85]
    settings['arr_support'] = 0.05   # [0..1]
    settings['arr_confidence'] = 10 # [50..85]
    optimizer = sim.Simod(settings)
    optimizer.execute_pipeline()

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
    settings = dict()
    column_names = {'Case ID': 'caseid', 'Activity': 'task',
                    'lifecycle:transition': 'event_type', 'Resource': 'user'}
    # Event-log reading options
    settings['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                      'column_names': column_names,
                                      'one_timestamp': False,
                                      'filter_d_attrib': True}
    # Folders structure
    settings['input'] = 'inputs'
    settings['output'] = os.path.join('outputs', sup.folder_id())
    # External tools routes
    settings['sm2_path'] = os.path.join('external_tools',
                                              'splitminer2',
                                              'sm2.jar')
    settings['sm1_path'] = os.path.join('external_tools',
                                              'splitminer',
                                              'splitminer.jar')
    settings['sm3_path'] = os.path.join('external_tools',
                                              'splitminer3',
                                              'bpmtk.jar')
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
    settings['mining_alg'] = 'sm3'
    return settings


if __name__ == "__main__":
    main(sys.argv[1:])

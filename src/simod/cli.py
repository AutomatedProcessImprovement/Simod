import os

import click
import utils.support as sup
from simod.discoverer import Discoverer
from simod.discovery_optimizer import DiscoveryOptimizer


@click.group()
def main():
    pass


@main.command()
@click.option('-l', '--log_path', required=True)
@click.option('-m', '--model_path', default=None, type=str)
@click.option('--mining_alg', default='sm3', show_default=True,
              type=click.Choice(['sm1', 'sm2', 'sm3'], case_sensitive=False))
@click.option('--alg_manag', default='repair', show_default=True,
              type=click.Choice(['replacement', 'repair', 'removal'], case_sensitive=False))
@click.option('--arr_confidence', default=9.2, show_default=True, type=float)
@click.option('--arr_support', default=0.098, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option('--epsilon', default=0.27747592754346484, show_default=True, type=float)
@click.option('--eta', default=0.3475132024591636, show_default=True, type=float)
@click.option('--gate_management', default='discovery', show_default=True,
              type=click.Choice(['discovery', 'equiprobable'], case_sensitive=False))
@click.option('--res_confidence', default=56.03564752634776, show_default=True, type=float)
@click.option('--res_support', default=0.07334543198001255, show_default=True,
              type=click.FloatRange(0.0, 1.0))
@click.option('--res_cal_met', default='discovered', show_default=True,
              type=click.Choice(['discovered', 'default'], case_sensitive=False))
@click.option('--res_dtype', default='247', show_default=True,
              type=click.Choice(['LV917', '247'], case_sensitive=False))
@click.option('--arr_dtype', default='247', show_default=True,
              type=click.Choice(['LV917', '247'], case_sensitive=False))
@click.option('--rp_similarity', default=0.8, show_default=True, type=float)
@click.option('--pdef_method', default='automatic', show_default=True,
              type=click.Choice(['manual', 'automatic', 'semi-automatic'], case_sensitive=False))
@click.pass_context
def discover(ctx, log_path, model_path, mining_alg, alg_manag, arr_confidence, arr_support,
             arr_dtype, epsilon, eta, gate_management, res_confidence, res_support, res_cal_met,
             res_dtype, rp_similarity, pdef_method):
    def define_general_settings(settings: dict = None) -> dict:
        """ Sets the app general settings"""
        if not settings:
            settings = dict()
        column_names = {'Case ID': 'caseid', 'Activity': 'task',
                        'lifecycle:transition': 'event_type', 'Resource': 'user'}
        # Event-log reading options
        settings['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                    'column_names': column_names,
                                    'one_timestamp': False,
                                    'filter_d_attrib': True}
        # Folders structure
        settings['output'] = os.path.join('outputs', sup.folder_id())
        # External tools routes
        settings['sm2_path'] = os.path.join('external_tools', 'splitminer2', 'sm2.jar')
        settings['sm1_path'] = os.path.join('external_tools', 'splitminer', 'splitminer.jar')
        settings['sm3_path'] = os.path.join('external_tools', 'splitminer3', 'bpmtk.jar')
        settings['bimp_path'] = os.path.join('external_tools', 'bimp', 'qbp-simulator-engine.jar')
        settings['align_path'] = os.path.join('external_tools', 'proconformance',
                                              'ProConformance2.jar')
        settings['aligninfo'] = os.path.join(settings['output'], 'CaseTypeAlignmentResults.csv')
        settings['aligntype'] = os.path.join(settings['output'], 'AlignmentStatistics.csv')
        settings['calender_path'] = os.path.join('external_tools', 'calenderimp', 'CalenderImp.jar')
        settings['simulator'] = 'bimp'
        settings['mining_alg'] = 'sm3'
        return settings

    if model_path is None:
        click.echo("Model is missing. It will be dynamically discovered from the log file.")

    settings = define_general_settings()
    settings['project_name'], _ = os.path.splitext(os.path.basename(log_path))
    settings['repetitions'] = 1
    settings['simulation'] = True
    settings['sim_metric'] = 'tsd'
    settings['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    settings['concurrency'] = 0.0
    settings['arr_cal_met'] = 'discovered'
    settings.update(ctx.params)

    optimizer = Discoverer(settings)
    optimizer.execute_pipeline()


@main.command()
@click.option('-l', '--log_path', required=True)
@click.option('--mining_alg', default='sm3', show_default=True,
              type=click.Choice(['sm1', 'sm2', 'sm3'], case_sensitive=False))
@click.pass_context
def optimize(ctx, log_path, mining_alg):
    def define_general_settings(settings: dict = None) -> dict:
        """ Sets the app general settings"""
        if not settings:
            settings = dict()
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
        settings['gl']['sm2_path'] = os.path.join('external_tools', 'splitminer2', 'sm2.jar')
        settings['gl']['sm1_path'] = os.path.join('external_tools', 'splitminer',
                                                  'splitminer.jar')
        settings['gl']['sm3_path'] = os.path.join('external_tools', 'splitminer3', 'bpmtk.jar')
        settings['gl']['bimp_path'] = os.path.join('external_tools', 'bimp',
                                                   'qbp-simulator-engine.jar')
        settings['gl']['align_path'] = os.path.join('external_tools', 'proconformance',
                                                    'ProConformance2.jar')
        settings['gl']['aligninfo'] = os.path.join(settings['gl']['output'],
                                                   'CaseTypeAlignmentResults.csv')
        settings['gl']['aligntype'] = os.path.join(settings['gl']['output'],
                                                   'AlignmentStatistics.csv')
        settings['gl']['calender_path'] = os.path.join('external_tools', 'calenderimp',
                                                       'CalenderImp.jar')
        settings['gl']['simulator'] = 'bimp'
        return settings

    settings = define_general_settings()
    settings['gl']['input'] = os.path.dirname(log_path)
    settings['gl']['file'] = os.path.basename(log_path)
    settings['gl']['file'] = os.path.basename(log_path)
    settings['gl']['mining_alg'] = mining_alg
    settings['gl']['exec_mode'] = 'optimizer'  # 'single', 'optimizer'
    settings['gl']['repetitions'] = 5
    settings['gl']['simulation'] = True
    settings['gl']['sim_metric'] = 'tsd'
    settings['gl']['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    settings['strc'] = dict()
    settings['strc']['max_eval_s'] = 15
    settings['strc']['concurrency'] = [0.0, 1.0]
    settings['strc']['epsilon'] = [0.0, 1.0]
    settings['strc']['eta'] = [0.0, 1.0]
    settings['strc']['alg_manag'] = ['replacement', 'repair', 'removal']
    settings['strc']['gate_management'] = ['discovery', 'equiprobable']
    settings['tm'] = dict()
    settings['tm']['max_eval_t'] = 20
    settings['tm']['rp_similarity'] = [0.5, 0.9]
    settings['tm']['res_dtype'] = ['LV917', '247']
    settings['tm']['arr_dtype'] = ['LV917', '247']
    settings['tm']['res_sup_dis'] = [0.01, 0.3]  # [0..1]
    settings['tm']['res_con_dis'] = [50, 85]  # [50..85]
    settings['tm']['arr_support'] = [0.01, 0.1]  # [0..1]
    settings['tm']['arr_confidence'] = [1, 10]  # [50..85]
    settings.update(ctx.params)
    optimizer = DiscoveryOptimizer(settings)
    optimizer.execute_pipeline()


if __name__ == "__main__":
    main()

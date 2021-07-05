import os

import click
import utils.support as sup
from simod.discoverer import Discoverer


@click.group()
def main():
    click.echo("main tool")


@main.command()
@click.option('-l', '--logfile', required=True, default='inputs/PurchasingExample.xes',
              show_default=True)
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
def discover(ctx, logfile, mining_alg, alg_manag, arr_confidence, arr_support, arr_dtype, epsilon,
             eta, gate_management, res_confidence, res_support, res_cal_met, res_dtype,
             rp_similarity, pdef_method):
    settings = define_general_settings()
    settings['repetitions'] = 1
    settings['simulation'] = True
    settings['sim_metric'] = 'tsd'
    settings['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    settings['concurrency'] = 0.0
    settings['arr_cal_met'] = 'discovered'
    settings['input'] = os.path.dirname(logfile)
    settings['file'] = os.path.basename(logfile)
    settings.update(ctx.params)
    optimizer = Discoverer(settings)
    optimizer.execute_pipeline()


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
    settings['align_path'] = os.path.join('external_tools', 'proconformance', 'ProConformance2.jar')
    settings['aligninfo'] = os.path.join(settings['output'], 'CaseTypeAlignmentResults.csv')
    settings['aligntype'] = os.path.join(settings['output'], 'AlignmentStatistics.csv')
    settings['calender_path'] = os.path.join('external_tools', 'calenderimp', 'CalenderImp.jar')
    settings['simulator'] = 'bimp'
    settings['mining_alg'] = 'sm3'
    return settings


@main.command()
def optimize():
    click.echo("hi optimizer")


if __name__ == "__main__":
    main()

import copy
import os
import pathlib
from pathlib import Path

import utils.support as sup
from simod.cli_formatter import *
from simod.configuration import Configuration, MiningAlgorithm, AlgorithmManagement, CalculationMethod, DataType, \
    PDFMethod, GateManagement, Metric, ExecutionMode
from simod.discoverer import Discoverer
from simod.optimizer import Optimizer
from simod.stochastic_miner import StochasticProcessMiner


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
@click.option('--gate_management', default='discovered', show_default=True,
              type=click.Choice(['discovered', 'equiprobable'], case_sensitive=False))
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
    ctx.params['mining_alg'] = MiningAlgorithm.from_str(mining_alg)
    ctx.params['alg_manag'] = AlgorithmManagement.from_str(alg_manag)
    ctx.params['gate_management'] = GateManagement.from_str(gate_management)
    ctx.params['res_cal_met'] = CalculationMethod.from_str(res_cal_met)
    ctx.params['res_dtype'] = DataType.from_str(res_dtype)
    ctx.params['arr_dtype'] = DataType.from_str(arr_dtype)
    ctx.params['pdef_method'] = PDFMethod.from_str(pdef_method)

    config = Configuration(**ctx.params)
    config.fill_in_derived_fields()

    discoverer = Discoverer(config)
    discoverer.execute_pipeline()


@main.command()
@click.option('-l', '--log_path', required=True)
@click.option('--mining_alg', default='sm3', show_default=True,
              type=click.Choice(['sm1', 'sm2', 'sm3'], case_sensitive=False))
@click.pass_context
def optimize(ctx, log_path, mining_alg):
    ctx.params['mining_alg'] = MiningAlgorithm.from_str(mining_alg)
    global_config = Configuration(input=Path('inputs'),
                                  output=Path(os.path.join('outputs', sup.folder_id())),
                                  exec_mode=ExecutionMode.OPTIMIZER,
                                  repetitions=1,
                                  # repetitions=5,
                                  simulation=True,
                                  sim_metric=Metric.DL,
                                  # sim_metric=Metric.TSD,
                                  add_metrics=[],
                                  # add_metrics=[Metric.DAY_HOUR_EMD,
                                  #              Metric.LOG_MAE, Metric.DL,
                                  #              Metric.MAE],
                                  **ctx.params)
    global_config.fill_in_derived_fields()

    structure_optimizer_config = copy.copy(global_config)
    structure_optimizer_config.max_eval_s = 2
    # structure_optimizer_config.max_eval_s = 15
    structure_optimizer_config.concurrency = [0.0, 1.0]
    structure_optimizer_config.epsilon = [0.0, 1.0]
    structure_optimizer_config.eta = [0.0, 1.0]
    structure_optimizer_config.alg_manag = [AlgorithmManagement.REMOVAL]
    # structure_optimizer_config.alg_manag = [AlgorithmManagement.REPAIR,
    #                                         AlgorithmManagement.REMOVAL,
    #                                         AlgorithmManagement.REPLACEMENT]
    structure_optimizer_config.gate_management = [GateManagement.EQUIPROBABLE]
    # structure_optimizer_config.gate_management = [GateManagement.DISCOVERY,
    #                                               GateManagement.EQUIPROBABLE]

    time_optimizer_config = copy.copy(global_config)
    time_optimizer_config.max_eval_t = 2
    # time_optimizer_config.max_eval_t = 20
    time_optimizer_config.rp_similarity = [0.5, 0.9]
    time_optimizer_config.res_dtype = [DataType.DT247]
    # [DataType.DT247, DataType.LV917]
    time_optimizer_config.arr_dtype = [DataType.DT247]
    # [DataType.DT247, DataType.LV917]
    time_optimizer_config.res_sup_dis = [0.01, 0.3]
    time_optimizer_config.res_con_dis = [50, 85]
    time_optimizer_config.arr_support = [0.01, 0.1]
    time_optimizer_config.arr_confidence = [1, 10]

    optimizer = Optimizer(
        {'gl': global_config, 'strc': structure_optimizer_config, 'tm': time_optimizer_config})
    optimizer.execute_pipeline()


@main.command()
@click.option('-l', '--log_path', required=True)
@click.option('-m', '--model_path')
@click.pass_context
def optimize_with_new_replayer(ctx, log_path, model_path):
    if model_path:
        ctx.params['model_path'] = pathlib.Path(model_path)
    if log_path:
        ctx.params['log_path'] = pathlib.Path(log_path)

    config = Configuration(input=Path('inputs'), output=Path(os.path.join('outputs', sup.folder_id())), **ctx.params)
    miner = StochasticProcessMiner(config)
    miner.execute_pipeline()


if __name__ == "__main__":
    main()

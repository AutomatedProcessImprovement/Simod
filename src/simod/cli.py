import copy
import os
from pathlib import Path

import click
import utils.support as sup
from simod.configuration import Configuration, AlgorithmManagement, DataType, \
    GateManagement, Metric, ExecutionMode, config_data_from_file
from simod.discoverer import Discoverer
from simod.optimizer import Optimizer
from simod.stochastic_miner import StructureOptimizerForStochasticProcessMiner


@click.group()
def main():
    # This is a main group which includes other commands specified below.
    pass


@main.command()
@click.option('--config_path', default=None, required=True)
@click.pass_context
def discover(ctx, config_path):
    repository_dir = os.path.join(os.path.dirname(__file__), '../../')
    ctx.params['config_path'] = Path(os.path.join(repository_dir, config_path))

    config_data = config_data_from_file(config_path)
    config_data.update(ctx.params)
    config = Configuration(**config_data)
    config.fill_in_derived_fields()

    discoverer = Discoverer(config)
    discoverer.execute_pipeline()


@main.command()
@click.option('--config_path', default=None, required=True)
@click.pass_context
def optimize(ctx, config_path):
    repository_dir = os.path.join(os.path.dirname(__file__), '../../')
    ctx.params['config_path'] = Path(os.path.join(repository_dir, config_path))

    config_data = config_data_from_file(config_path)
    config_data.update(ctx.params)

    strc_data = config_data.pop('strc')
    tm_data = config_data.pop('tm')
    global_data = config_data

    global_config = Configuration(**global_data)
    global_config.fill_in_derived_fields()

    strc_data.update(global_data)
    structure_optimizer_config = Configuration(**strc_data)
    structure_optimizer_config.fill_in_derived_fields()

    tm_data.update(global_data)
    time_optimizer_config = Configuration(**tm_data)
    time_optimizer_config.fill_in_derived_fields()


    # global_config = Configuration(**config_data)
    # global_config.fill_in_derived_fields()
    #
    # global_config = Configuration(input=Path(os.path.join(repository_dir, 'inputs')),
    #                               output=Path(os.path.join(repository_dir, 'outputs', sup.folder_id())),
    #                               exec_mode=ExecutionMode.OPTIMIZER,
    #                               repetitions=1,
    #                               # repetitions=5,
    #                               simulation=True,
    #                               sim_metric=Metric.DL,
    #                               # sim_metric=Metric.TSD,
    #                               add_metrics=[],
    #                               # add_metrics=[Metric.DAY_HOUR_EMD,
    #                               #              Metric.LOG_MAE, Metric.DL,
    #                               #              Metric.MAE],
    #                               **ctx.params)
    # global_config.fill_in_derived_fields()

    # structure_optimizer_config = copy.copy(global_config)
    # structure_optimizer_config.max_eval_s = 2
    # # structure_optimizer_config.max_eval_s = 15
    # structure_optimizer_config.concurrency = [0.0, 1.0]
    # structure_optimizer_config.epsilon = [0.0, 1.0]
    # structure_optimizer_config.eta = [0.0, 1.0]
    # structure_optimizer_config.alg_manag = [AlgorithmManagement.REMOVAL]
    # # structure_optimizer_config.alg_manag = [AlgorithmManagement.REPAIR,
    # #                                         AlgorithmManagement.REMOVAL,
    # #                                         AlgorithmManagement.REPLACEMENT]
    # structure_optimizer_config.gate_management = [GateManagement.EQUIPROBABLE]
    # # structure_optimizer_config.gate_management = [GateManagement.DISCOVERY,
    # #                                               GateManagement.EQUIPROBABLE]

    # time_optimizer_config = copy.copy(global_config)
    # time_optimizer_config.max_eval_t = 2
    # # time_optimizer_config.max_eval_t = 20
    # time_optimizer_config.rp_similarity = [0.5, 0.9]
    # time_optimizer_config.res_dtype = [DataType.DT247]
    # # [DataType.DT247, DataType.LV917]
    # time_optimizer_config.arr_dtype = [DataType.DT247]
    # # [DataType.DT247, DataType.LV917]
    # time_optimizer_config.res_sup_dis = [0.01, 0.3]
    # time_optimizer_config.res_con_dis = [50, 85]
    # time_optimizer_config.arr_support = [0.01, 0.1]
    # time_optimizer_config.arr_confidence = [1, 10]

    optimizer = Optimizer({'gl': global_config, 'strc': structure_optimizer_config, 'tm': time_optimizer_config})
    if global_config.new_replayer:
        optimizer.execute_pipeline(
            structure_optimizer=StructureOptimizerForStochasticProcessMiner,
            discover_model=global_config.model_path is None)
    else:
        optimizer.execute_pipeline()


# @main.command()
# @click.option('-l', '--log_path', required=True)
# @click.option('-m', '--model_path', required=True)
# @click.pass_context
# def optimize_with_new_replayer(ctx, log_path, model_path):
#     ctx.params['model_path'] = pathlib.Path(model_path)
#     ctx.params['log_path'] = pathlib.Path(log_path)
#
#     config = Configuration(input=Path('inputs'), output=Path(os.path.join('outputs', sup.folder_id())), **ctx.params)
#     config.fill_in_derived_fields()
#     miner = StochasticProcessMiner(config)
#     miner.execute_pipeline()


if __name__ == "__main__":
    main()

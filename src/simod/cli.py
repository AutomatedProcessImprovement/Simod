import os
from pathlib import Path

import click
from simod.configuration import Configuration, config_data_from_file
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

    optimizer = Optimizer({'gl': global_config, 'strc': structure_optimizer_config, 'tm': time_optimizer_config})
    if global_config.new_replayer:
        optimizer.execute_pipeline(
            structure_optimizer=StructureOptimizerForStochasticProcessMiner,
            discover_model=global_config.model_path is None)
    else:
        optimizer.execute_pipeline()


if __name__ == "__main__":
    main()

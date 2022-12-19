from pathlib import Path

import click

from simod.configuration import Configuration
from simod.event_log.preprocessor import Preprocessor
from simod.optimization.optimizer import Optimizer
from simod.utilities import get_project_dir, folder_id


@click.group()
def main():
    # This is a main group which includes other commands specified below.
    pass


# @main.command()
# @click.option('--config_path', default=None, required=True, type=Path)
# @click.pass_context
# def discover(ctx, config_path):
#     repository_dir = get_project_dir()
#     ctx.params['config_path'] = repository_dir.joinpath(config_path)
#
#     config_data = config_data_from_file(config_path)
#     config_data.update(ctx.params)
#     config = Configuration(**config_data)
#
#     discoverer = Discoverer(config)
#     discoverer.run()


@main.command()
@click.option('--config_path', default=None, required=True, type=Path)
@click.pass_context
def optimize(ctx, config_path):
    repository_dir = get_project_dir()
    config_path = repository_dir / config_path
    settings = Configuration.from_path(config_path)

    output_dir = get_project_dir() / 'outputs' / folder_id()

    preprocessor = Preprocessor(settings, output_dir)
    settings, event_log = preprocessor.run()

    Optimizer(settings, event_log=event_log, output_dir=output_dir).run()


if __name__ == "__main__":
    main()

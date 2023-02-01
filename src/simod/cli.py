from pathlib import Path

import click

from simod.configuration import Configuration
from simod.event_log.event_log import EventLog
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.utilities import read
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
@click.option('--config_path', default=None, required=True, type=str)
@click.option('--output_dir', default=None, required=False, type=str)
def optimize(config_path: str, output_dir: str) -> Path:
    repository_dir = get_project_dir()
    config_path = repository_dir / config_path
    settings = Configuration.from_path(config_path)

    output_dir = Path(output_dir) if output_dir is not None else None

    # NOTE: EventLog requires start_time column to be present for split_log() to work.
    #   So, we do pre-processing before creating the EventLog object.

    log, csv_path = read(settings.common.log_path, settings.common.log_ids)

    preprocessor = Preprocessor(log, settings.common.log_ids)
    processed_log = preprocessor.run(
        multitasking=settings.preprocessing.multitasking,
    )

    test_log = None
    if settings.common.test_log_path is not None:
        test_log, _ = read(settings.common.test_log_path, settings.common.log_ids)

    event_log = EventLog.from_df(
        log=processed_log,  # would be split into training and validation if test is provided, otherwise into test too
        log_ids=settings.common.log_ids,
        process_name=settings.common.log_path.stem,
        test_log=test_log,
        log_path=settings.common.log_path,
        csv_log_path=csv_path,
    )

    if output_dir is None:
        output_dir = get_project_dir() / 'outputs' / folder_id()

    Optimizer(settings, event_log=event_log, output_dir=output_dir).run()

    return output_dir


if __name__ == "__main__":
    main()

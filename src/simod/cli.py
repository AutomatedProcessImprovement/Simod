from pathlib import Path

import click
from pix_framework.filesystem.file_manager import get_random_folder_id

from simod.event_log.event_log import EventLog
from simod.settings.simod_settings import SimodSettings, PROJECT_DIR
from simod.simod import Simod


@click.group()
def main():
    # This is a main group which includes other commands specified below.
    pass


@main.command()
@click.option("--config_path", default=None, required=True, type=str)
@click.option("--output_dir", default=None, required=False, type=str)
def optimize(config_path: str, output_dir: str) -> Path:
    # Read configuration file
    config_path = PROJECT_DIR / config_path
    settings = SimodSettings.from_path(config_path)
    # Instantiate output directory path if specified
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_DIR / "outputs" / get_random_folder_id()
    # Read and preprocess event log
    event_log = EventLog.from_path(
        train_log_path=settings.common.train_log_path,
        log_ids=settings.common.log_ids,
        test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
        need_test_partition=settings.common.perform_final_evaluation,
    )
    # Instantiate and run Simod
    simod = Simod(settings, event_log=event_log, output_dir=output_dir)
    simod.run()
    # Return output directory
    return output_dir


if __name__ == "__main__":
    main()

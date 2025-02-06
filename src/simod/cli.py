import json
from pathlib import Path
from typing import Optional

import click
import yaml
from pix_framework.filesystem.file_manager import get_random_folder_id

from simod.event_log.event_log import EventLog
from simod.runtime_meter import RuntimeMeter
from simod.settings.simod_settings import SimodSettings
from simod.simod import Simod


@click.command(
    help="""
    Simod combines process mining and machine learning techniques to automate the discovery and tuning of
    Business Process Simulation models from event logs extracted from enterprise information systems.
    """
)
@click.option(
    "--configuration",
    "-c",
    default=None,
    required=False,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Path to the Simod configuration file.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    required=False,
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    help="Path to the output directory where discovery results will be stored.",
)
@click.option(
    "--one-shot",
    default=False,
    is_flag=True,
    required=False,
    type=bool,
    help="Run Simod with default settings only once without the optimization phase.",
)
@click.option(
    "--event-log",
    "-l",
    required=False,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Path to the event log file when using the --one-shot flag. "
    "Columns must be named 'case_id', 'activity', 'start_time', 'end_time', 'resource'.",
)
@click.option(
    "--schema-yaml",
    required=False,
    is_flag=True,
    help="Print the configuration YAML schema and exit.",
)
@click.option(
    "--schema-json",
    required=False,
    is_flag=True,
    help="Print the configuration JSON schema and exit.",
)
@click.version_option()
def main(
    configuration: Optional[Path],
    output: Optional[Path],
    one_shot: bool,
    event_log: Optional[Path],
    schema_yaml: bool,
    schema_json: bool,
) -> None:
    if schema_yaml:
        print(yaml.dump(SimodSettings().model_json_schema()))
        return

    if schema_json:
        print(json.dumps(SimodSettings().model_json_schema()))
        return

    if one_shot:
        settings = SimodSettings.one_shot()
        settings.common.train_log_path = event_log
        settings.common.test_log_path = None
    else:
        settings = SimodSettings.from_path(configuration)

    output = output if output is not None else (Path.cwd() / "outputs" / get_random_folder_id()).absolute()

    # To measure the runtime of each stage
    runtimes = RuntimeMeter()

    # Read and preprocess event log
    runtimes.start(RuntimeMeter.PREPROCESSING)
    event_log = EventLog.from_path(
        log_ids=settings.common.log_ids,
        train_log_path=settings.common.train_log_path,
        test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
        need_test_partition=settings.common.perform_final_evaluation,
    )
    runtimes.stop(RuntimeMeter.PREPROCESSING)

    # Instantiate and run Simod
    simod = Simod(settings, event_log=event_log, output_dir=output)
    simod.run(runtimes=runtimes)


if __name__ == "__main__":
    main()

from pathlib import Path

import pytest
from pix_utils.filesystem.file_manager import get_random_folder_id

from simod import cli
from simod.event_log.event_log import EventLog
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.utilities import read
from simod.optimization.optimizer import Optimizer
from simod.settings.simod_settings import PROJECT_DIR, SimodSettings

# NOTE: these are mostly general overall long-running tests to check if everything finishes without exceptions

optimize_config_files = [
    'optimize_config_no_start_times.yml',
    # 'optimize_config_with_timers.yml',
]


@pytest.mark.system
@pytest.mark.parametrize('path', optimize_config_files)
def test_optimize(entry_point, runner, path):
    config_path = entry_point / path
    result = runner.invoke(cli.main, ['optimize', '--config_path', config_path.absolute()])
    assert not result.exception
    assert result.exit_code == 0


def test_SIMOD():
    config_path = Path("./tests/assets/optimize_config_no_start_times.yml")
    settings = SimodSettings.from_path(config_path)

    output_dir = None

    log, csv_path = read(settings.common.log_path, settings.common.log_ids)

    preprocessor = Preprocessor(log, settings.common.log_ids)
    processed_log = preprocessor.run(
        multitasking=settings.preprocessing.multitasking,
        enable_time_concurrency_threshold=settings.preprocessing.enable_time_concurrency_threshold,
        concurrency_thresholds=settings.preprocessing.concurrency_thresholds
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
        output_dir = PROJECT_DIR / 'outputs' / get_random_folder_id()

    Optimizer(settings, event_log=event_log, output_dir=output_dir).run()

    assert True
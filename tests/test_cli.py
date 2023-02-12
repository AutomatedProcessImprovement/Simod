import pytest

from simod import cli

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

import pytest

from simod import cli

# NOTE: these are mostly general overall long-running tests to check if everything finishes without exceptions

discover_config_files = [
    'discover_without_model_config.yml',
    'discover_with_model_config.yml',
]

optimize_config_files = [
    'optimize_debug_config.yml',
    'optimize_debug_with_model_config.yml',
    'optimize_debug_config_2.yml',
]


# @pytest.mark.integration
# @pytest.mark.parametrize('path', discover_config_files)
# def test_discover(entry_point, runner, path):
#     config_path = os.path.join(entry_point, path)
#     assert os.path.exists(config_path)
#     result = runner.invoke(cli.main, ['discover', '--config_path', config_path])
#     assert not result.exception
#     assert result.exit_code == 0


@pytest.mark.integration
@pytest.mark.parametrize('path', optimize_config_files)
def test_optimize(entry_point, runner, path):
    config_path = entry_point / path
    print(f'\nConfig file: {config_path}')
    result = runner.invoke(cli.main, ['optimize', '--config_path', config_path])
    assert not result.exception
    assert result.exit_code == 0

import os

import pytest

from simod.cli import main

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


@pytest.mark.slow
def test_discover(entry_point, runner):
    for path in discover_config_files:
        config_path = os.path.join(entry_point, path)
        result = runner.invoke(main, ['discover', '--config_path', config_path])
        assert not result.exception
        assert result.exit_code == 0


@pytest.mark.slow
def test_optimize(entry_point, runner):
    for path in optimize_config_files:
        config_path = os.path.join(entry_point, path)
        result = runner.invoke(main, ['optimize', '--config_path', config_path])
        assert not result.exception
        assert result.exit_code == 0

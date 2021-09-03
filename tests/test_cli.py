import os

from simod.cli import main


def test_optimizer_without_model_without_andor_attribute(entry_point, runner):
    config_path = os.path.join(entry_point, 'optimize_debug_config_2.yml')
    result = runner.invoke(main, ['optimize', '--config_path', config_path])
    assert not result.exception
    assert result.exit_code == 0

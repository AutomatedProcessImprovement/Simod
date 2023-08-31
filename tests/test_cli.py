import pytest

from simod import cli


@pytest.mark.system
@pytest.mark.parametrize("path", ["configuration_simod_basic.yml"])
def test_optimize(entry_point, runner, path):
    config_path = entry_point / path
    result = runner.invoke(cli.main, ["--configuration", config_path.absolute()])
    assert not result.exception
    assert result.exit_code == 0

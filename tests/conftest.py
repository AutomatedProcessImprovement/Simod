import os
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture(scope='function')
def runner(request):
    return CliRunner()


@pytest.fixture(scope='module')
def entry_point() -> Path:
    if Path.cwd().name == 'tests':
        return Path('assets')
    else:
        return Path('tests/assets')

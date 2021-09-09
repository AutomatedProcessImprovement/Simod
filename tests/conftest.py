import os

import pytest
from click.testing import CliRunner


@pytest.fixture(scope='function')
def runner(request):
    return CliRunner()


@pytest.fixture(scope='module')
def entry_point():
    if os.path.basename(os.getcwd()) == 'tests':
        return 'assets'
    else:
        return 'tests/assets'

import os
from pathlib import Path

import pytest

from simod.configuration import Configuration
from simod.simulator import diffresbp_simulator, get_number_of_cases


@pytest.fixture
def args(entry_point, tmp_path) -> list:
    entry_point = Path(entry_point)
    return [
        {'qbp_path': entry_point / 'Production.bpmn',
         'json_path': tmp_path / 'Production.json',
         'n_cases': 45},
        {'qbp_path': entry_point / 'PurchasingExampleQBP.bpmn',
         'json_path': tmp_path / 'PurchasingExampleQBP.json',
         'n_cases': 98},
    ]


def test_diffresbp_simulator(args):
    for arg in args:
        config = Configuration()
        config.output = arg['qbp_path'].parent
        config.project_name, _ = os.path.splitext(arg['qbp_path'].name)
        config.repetitions = 1
        config.simulation_cases = get_number_of_cases(arg['qbp_path'])

        diffresbp_simulator((config, config.repetitions))

        json_path = arg['qbp_path'].with_suffix('.json')
        assert json_path.exists()


# def test_simulate(args):
#     settings = Configuration()
#     stats = pd.DataFrame()
#     log_data = pd.DataFrame()
#
#     simulate(settings, stats, log_data, evaluate_fn=evaluate_logs)


def test_get_number_of_cases(args):
    for arg in args:
        n_cases = get_number_of_cases(arg['qbp_path'])
        assert arg['n_cases'] == n_cases

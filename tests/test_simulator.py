import os
import shutil

import pytest

from simod.configuration import Configuration
from simod.simulator import diffresbp_simulator, get_number_of_cases

test_data = [
    {'qbp_path': 'Production.bpmn', 'n_cases': 45},
    {'qbp_path': 'PurchasingExampleQBP.bpmn', 'n_cases': 98},
]


@pytest.mark.parametrize('arg', test_data, ids=map(lambda x: x['qbp_path'], test_data))
def test_diffresbp_simulator(entry_point, arg):
    qbp_path = entry_point / arg['qbp_path']

    config = Configuration()
    config.output = qbp_path.parent
    config.project_name, _ = os.path.splitext(qbp_path.name)
    config.repetitions = 1
    config.simulation_cases = get_number_of_cases(qbp_path)

    diffresbp_simulator((config, config.repetitions))

    json_path = qbp_path.with_suffix('.json')
    assert json_path.exists()

    output_dir = config.output / 'sim_data'
    shutil.rmtree(output_dir)


@pytest.mark.parametrize('arg', test_data, ids=map(lambda x: f'n_cases={x["n_cases"]}', test_data))
def test_get_number_of_cases(entry_point, arg):
    qbp_path = entry_point / arg['qbp_path']

    n_cases = get_number_of_cases(qbp_path)
    assert arg['n_cases'] == n_cases

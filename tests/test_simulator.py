import os
from pathlib import Path

import pytest

from bpdfr_simulation_engine.simulation_properties_parser import parse_qbp_simulation_process
from simod.configuration import Configuration
from simod.simulator import qbp_simulator, diffresbp_simulator, get_number_of_cases


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


def test_qbp_parse(args):
    for arg in args:
        parse_qbp_simulation_process(arg['qbp_path'], arg['json_path'])
        print(f"\nJSON saved to {arg['json_path']}")
        assert arg['json_path'].exists()


def test_qbp_simulator(args):
    for arg in args:
        config = Configuration()
        config.output = arg['qbp_path'].parent
        config.project_name, _ = os.path.splitext(arg['qbp_path'].name)
        config.repetitions = 1
        sim_data_path = config.output / 'sim_data'  # required tmp dir
        sim_data_path.mkdir(exist_ok=True)
        qbp_simulator((config, config.repetitions))

        output_files = list(sim_data_path.iterdir())
        print(output_files)
        assert len(output_files) != 0

        # tmp clean up
        for f in output_files:
            f.unlink()
        sim_data_path.rmdir()


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

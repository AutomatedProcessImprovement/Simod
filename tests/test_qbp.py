import os
from pathlib import Path

import pytest

from simod.configuration import Configuration
from simod.qbp import parse_model_and_save_json, execute_simulator


@pytest.fixture
def args(entry_point, tmp_path) -> list:
    entry_point = Path(entry_point)
    return [
        {'qbp_path': entry_point / 'Production.bpmn', 'json_path': tmp_path / 'Production.json'},
        {'qbp_path': entry_point / 'PurchasingExampleQBP.bpmn', 'json_path': tmp_path / 'PurchasingExampleQBP.json'},
    ]


def test_qbp_parse(args):
    for arg in args:
        parse_model_and_save_json(arg['qbp_path'], arg['json_path'])
        print(f"\nJSON saved to {arg['json_path']}")
        assert arg['json_path'].exists()


def test_execute_simulator(args):
    for arg in args:
        config = Configuration()
        config.output = arg['qbp_path'].parent
        config.project_name, _ = os.path.splitext(arg['qbp_path'].name)
        config.repetitions = 1
        sim_data_path = config.output / 'sim_data'  # required tmp dir
        sim_data_path.mkdir(exist_ok=True)
        execute_simulator((config, config.repetitions))

        output_files = list(sim_data_path.iterdir())
        print(output_files)
        assert len(output_files) != 0

        # tmp clean up
        for f in output_files:
            f.unlink()
        sim_data_path.rmdir()

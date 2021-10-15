from pathlib import Path

import pytest

from simod.qbp import parse_and_save


@pytest.fixture
def args(entry_point, tmp_path) -> list:
    entry_point = Path(entry_point)
    return [
        {'qbp_path': entry_point / 'Production.bpmn', 'json_path': tmp_path / 'Production.json'},
        {'qbp_path': entry_point / 'PurchasingExampleQBP.bpmn', 'json_path': tmp_path / 'PurchasingExampleQBP.json'},
    ]


def test_qbp_parse(args):
    for arg in args:
        parse_and_save(arg['qbp_path'], arg['json_path'])
        print(f"\nJSON saved to {arg['json_path']}")
        assert arg['json_path'].exists()

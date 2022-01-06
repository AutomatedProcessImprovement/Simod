from pathlib import Path

from simod.multitasking import multitasking as mt


def test_pre_sweeper(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    mt.pre_sweeper(log_path)
    assert log_output_path.exists()


def test_apply_percentage(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    tree = mt.pre_sweeper(log_path)
    result = mt.apply_percentage(tree)
    assert result


def test_apply_sweeper(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    tree = mt.pre_sweeper(log_path)
    result = mt.apply_percentage(tree)
    result = mt.apply_sweeper(result)
    assert result

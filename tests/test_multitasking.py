from pathlib import Path

from simod.multitasking import multitasking as mt


def test_pre_sweeper(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    mt._pre_sweeper(log_path)


def test_apply_percentage(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    tree = mt._pre_sweeper(log_path)
    result = mt._apply_percentage(tree)
    assert result


def test_apply_sweeper(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    tree = mt._pre_sweeper(log_path)
    result = mt._apply_percentage(tree)
    result = mt._apply_sweeper(result)
    assert result

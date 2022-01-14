from pathlib import Path

from simod.multitasking import multitasking_alt as mt


def test_intro(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample.xes'
    result = mt._make_auxiliary_log(log_path)


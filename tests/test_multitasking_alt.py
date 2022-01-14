from pathlib import Path

from simod.multitasking import multitasking_alt as mt


def test_adjust_durations_purchasing_example(entry_point):
    log_path = Path(entry_point) / 'PurchasingExampleMultitasking.xes'
    result = mt.adjust_durations(log_path, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 330.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 870.0


def test_adjust_durations_other(entry_point):
    log_path = Path(entry_point) / 'ConsultaDataMining201618Multitasking.xes'
    # log_path = Path(entry_point) / 'APreproc_ConsultaDataMining201618Multitasking.xes'
    result = mt.adjust_durations(log_path, verbose=False)
    assert result is not None

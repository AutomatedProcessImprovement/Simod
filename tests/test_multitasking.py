from pathlib import Path

from simod.multitasking import adjust_durations
from simod.event_log import read, reformat_timestamps


def test_adjust_durations_purchasing_example(entry_point):
    log_path = Path(entry_point) / 'PurchasingExampleMultitasking.xes'
    log = read(log_path)
    result = adjust_durations(log, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 330.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 870.0


def test_adjust_durations_purchasing_example2(entry_point):
    log_path = Path(entry_point) / 'PurchasingExampleMultitasking2.xes'
    log = read(log_path)
    result = adjust_durations(log, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 600.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 1140.0


def test_adjust_durations_purchasing_example3(entry_point):
    log_path = Path(entry_point) / 'PurchasingExampleMultitasking3.xes'
    log = read(log_path)
    result = adjust_durations(log, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 5.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 2.5
    assert (result.iloc[2]['time:timestamp'] - result.iloc[2]['start_timestamp']).total_seconds() == 2.5


def test_adjust_durations_consulta(entry_point):
    log_path = Path(entry_point) / 'ConsultaDataMining201618.xes'
    log = read(log_path)
    result = adjust_durations(log, verbose=False)
    assert result is not None


def test_reformat_timestamps(entry_point):
    log_path = Path(entry_point) / 'PurchasingExample_Timestamps.xes'
    output_path = log_path.with_name(log_path.stem + '_Reformatted.xes')
    reformat_timestamps(log_path, output_path)
    assert output_path.exists()

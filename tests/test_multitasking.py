from pathlib import Path

import pytest

from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.multitasking import adjust_durations
from simod.event_log.utilities import read, reformat_timestamps


@pytest.mark.integration
def test_adjust_durations_purchasing_example(entry_point):
    log_path = Path(entry_point) / 'PurchasingExampleMultitasking.xes'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, STANDARD_COLUMNS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 330.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 870.0
    log_path_csv.unlink()


@pytest.mark.integration
def test_adjust_durations_purchasing_example2(entry_point):
    log_path = entry_point / 'PurchasingExampleMultitasking2.xes'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, STANDARD_COLUMNS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 600.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 1140.0
    log_path_csv.unlink()


@pytest.mark.integration
def test_adjust_durations_purchasing_example3(entry_point):
    log_path = entry_point / 'PurchasingExampleMultitasking3.xes'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, STANDARD_COLUMNS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 5.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 2.5
    assert (result.iloc[2]['time:timestamp'] - result.iloc[2]['start_timestamp']).total_seconds() == 2.5
    log_path_csv.unlink()


@pytest.mark.integration
def test_adjust_durations_consulta(entry_point):
    log_path = entry_point / 'ConsultaDataMining201618.xes'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, STANDARD_COLUMNS, verbose=False)
    assert result is not None
    log_path_csv.unlink()


def test_reformat_timestamps(entry_point):
    log_path = entry_point / 'PurchasingExample_Timestamps.xes'
    output_path = log_path.with_name(log_path.stem + '_Reformatted.xes')
    reformat_timestamps(log_path, output_path)
    assert output_path.exists()
    output_path.unlink()

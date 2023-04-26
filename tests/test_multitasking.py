import pytest
from pix_utils.log_ids import DEFAULT_XES_IDS

from simod.event_log.multitasking import adjust_durations
from simod.event_log.utilities import read, reformat_timestamps


@pytest.mark.integration
def test_adjust_durations_purchasing_example(entry_point):
    log_path = entry_point / 'PurchasingExampleMultitasking.csv'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 330.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 870.0


@pytest.mark.integration
def test_adjust_durations_purchasing_example2(entry_point):
    log_path = entry_point / 'PurchasingExampleMultitasking2.csv'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 600.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 1140.0


@pytest.mark.integration
def test_adjust_durations_purchasing_example3(entry_point):
    log_path = entry_point / 'PurchasingExampleMultitasking3.xes'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=True)
    assert result is not None
    assert (result.iloc[0]['time:timestamp'] - result.iloc[0]['start_timestamp']).total_seconds() == 5.0
    assert (result.iloc[1]['time:timestamp'] - result.iloc[1]['start_timestamp']).total_seconds() == 2.5
    assert (result.iloc[2]['time:timestamp'] - result.iloc[2]['start_timestamp']).total_seconds() == 2.5


@pytest.mark.integration
def test_adjust_durations_consulta(entry_point):
    log_path = entry_point / 'ConsultaDataMining201618.csv'
    log, log_path_csv = read(log_path)
    result = adjust_durations(log, DEFAULT_XES_IDS, verbose=False)
    assert result is not None


def test_reformat_timestamps(entry_point):
    log_path = entry_point / 'PurchasingExample_Timestamps.xes'
    output_path = log_path.with_name(log_path.stem + '_Reformatted.xes')
    reformat_timestamps(log_path, output_path)
    assert output_path.exists()
    output_path.unlink()

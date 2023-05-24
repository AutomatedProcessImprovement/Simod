import pytest
from pix_framework.input import read_csv_log
from pix_framework.log_ids import DEFAULT_CSV_IDS, APROMORE_LOG_IDS

from simod.metrics import get_absolute_hourly_emd

test_cases = [
    {
        'name': 'A',
        'event_log_1': {
            'log_name': 'LoanApp_sequential_9-5_diffres_timers.csv',
            'log_ids': DEFAULT_CSV_IDS
        },
        'event_log_2': {
            'log_name': 'simulated_log_0.csv',
            'log_ids': APROMORE_LOG_IDS
        },
    }
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_absolute_timestamp_emd(entry_point, test_data):
    event_log_1_path = entry_point / test_data['event_log_1']['log_name']
    event_log_2_path = entry_point / test_data['event_log_2']['log_name']

    event_log_1_log_ids = test_data['event_log_1']['log_ids']
    event_log_2_log_ids = test_data['event_log_2']['log_ids']

    event_log_1 = read_csv_log(event_log_1_path, event_log_1_log_ids)
    event_log_2 = read_csv_log(event_log_2_path, event_log_2_log_ids)

    emd = get_absolute_hourly_emd(event_log_1, event_log_1_log_ids, event_log_2, event_log_2_log_ids)

    assert emd > 0

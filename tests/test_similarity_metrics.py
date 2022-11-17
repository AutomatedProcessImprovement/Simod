import pytest

from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.utilities import read
from simod.metrics.metrics import get_absolute_hourly_emd

test_cases = [
    {
        'name': 'A',
        'event_log_1': {
            'log_name': 'LoanApp_sequential_9-5.csv',
            'log_ids': EventLogIDs(
                resource='org:resource',
                activity='concept:name',
                start_time='start_timestamp',
                end_time='time:timestamp',
                case='case:concept:name',
            )
        },
        'event_log_2': {
            'log_name': 'simulated_log_0.csv',
            'log_ids': EventLogIDs(
                resource='resource',
                activity='activity',
                start_time='start_time',
                end_time='end_time',
                case='case_id',
                enabled_time='enabled_time',
            )
        },
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_absolute_timestamp_emd(entry_point, test_data):
    event_log_1_path = entry_point / test_data['event_log_1']['log_name']
    event_log_2_path = entry_point / test_data['event_log_2']['log_name']

    event_log_1_log_ids = test_data['event_log_1']['log_ids']
    event_log_2_log_ids = test_data['event_log_2']['log_ids']

    event_log_1, _ = read(event_log_1_path, event_log_1_log_ids)
    event_log_2, _ = read(event_log_2_path, event_log_2_log_ids)

    emd = get_absolute_hourly_emd(event_log_1, event_log_1_log_ids, event_log_2, event_log_2_log_ids)

    assert emd > 0

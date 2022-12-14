import pytest

from simod.discovery import inter_arrival_distribution
from simod.event_log.column_mapping import STANDARD_COLUMNS
from simod.event_log.utilities import read

test_cases = [
    {
        'name': 'A',
        'log_name': 'LoanApp_sequential_9-5_diffres_filtered.csv',
        'model_name': 'LoanApp_sequential_9-5_diffres_filtered.bpmn',
    }
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_discover(test_data, entry_point):
    log_path = entry_point / test_data['log_name']

    log_ids = STANDARD_COLUMNS
    (log, _) = read(log_path, log_ids)

    result = inter_arrival_distribution.discover(log, log_ids)

    assert result is not None

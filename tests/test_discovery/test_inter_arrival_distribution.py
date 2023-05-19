import pytest
from pix_framework.discovery.case_arrival import discover_inter_arrival_distribution
from pix_framework.log_ids import DEFAULT_XES_IDS

from simod.event_log.utilities import read

test_cases = [
    {
        'name': 'A',
        'log_name': 'LoanApp_sequential_9-5_diffres_filtered.csv',
        'model_name': 'LoanApp_sequential_9-5_diffres_filtered.bpmn',
    }
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_discover(test_data, entry_point):
    log_path = entry_point / test_data['log_name']

    log_ids = DEFAULT_XES_IDS
    log, _ = read(log_path, log_ids)

    result = discover_inter_arrival_distribution(log, log_ids)

    assert result is not None

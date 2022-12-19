import pytest

from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS
from simod.event_log.event_log import EventLog

test_cases = [
    {
        'log_name': 'LoanApp_sequential_9-5_diffres_timers.csv',
        'log_ids': EventLogIDs(
            case='case_id',
            activity='Activity',
            resource='Resource',
            start_time='start_time',
            end_time='end_time',
        )
    },
    {
        'log_name': 'Production.xes',
        'log_ids': STANDARD_COLUMNS,
    },
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['log_name'] for test_data in test_cases])
def test_optimizer(test_data, entry_point):
    path = (entry_point / test_data['log_name']).absolute()
    log_ids = test_data['log_ids']

    event_log = EventLog.from_path(path, log_ids)

    assert event_log.log_ids == log_ids
    assert event_log.log_path == path
    assert event_log.train_partition is not None
    assert event_log.validation_partition is not None
    assert event_log.test_partition is not None
    assert len(event_log.train_partition) > len(event_log.validation_partition)
    assert len(event_log.validation_partition) < len(event_log.test_partition)

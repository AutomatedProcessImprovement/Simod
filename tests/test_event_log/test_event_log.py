import pytest
from pix_framework.io.event_log import APROMORE_LOG_IDS, DEFAULT_XES_IDS
from simod.event_log.event_log import EventLog

test_cases = [
    {"log_name": "Simple_log_no_start_times.csv", "log_ids": APROMORE_LOG_IDS},
    {
        "log_name": "LoanApp_simplified.csv",
        "log_ids": DEFAULT_XES_IDS,
    },
]


@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["log_name"] for test_data in test_cases])
def test_optimizer(test_data, entry_point):
    path = (entry_point / test_data["log_name"]).absolute()
    log_ids = test_data["log_ids"]

    event_log = EventLog.from_path(path, log_ids)

    assert event_log.log_ids == log_ids
    assert event_log.train_partition is not None
    assert event_log.validation_partition is not None
    assert event_log.test_partition is not None
    assert len(event_log.train_partition) > len(event_log.validation_partition)
    assert len(event_log.validation_partition) < len(event_log.test_partition)

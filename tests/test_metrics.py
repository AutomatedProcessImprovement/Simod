import pytest
from pix_framework.io.event_log import DEFAULT_XES_IDS, read_csv_log
from simod.metrics import get_absolute_emd

test_cases = [
    {
        "name": "LoanApp_simplified",
        "original_log": {"log_name": "LoanApp_simplified.csv.gz", "log_ids": DEFAULT_XES_IDS},
        "simulated_log": {"log_name": "LoanApp_simplified_2.csv.gz", "log_ids": DEFAULT_XES_IDS},
    }
]


@pytest.mark.integration
@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["name"] for test_data in test_cases])
def test_absolute_timestamp_emd(entry_point, test_data):
    original_log_path = entry_point / test_data["original_log"]["log_name"]
    simulated_log_path = entry_point / test_data["simulated_log"]["log_name"]

    original_log_ids = test_data["original_log"]["log_ids"]
    simulated_log_ids = test_data["simulated_log"]["log_ids"]

    original_log = read_csv_log(original_log_path, original_log_ids)
    simulated_log = read_csv_log(simulated_log_path, simulated_log_ids)

    # Test different logs
    emd = get_absolute_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    assert emd > 0.0
    # Test similar log
    emd = get_absolute_emd(original_log, original_log_ids, original_log, simulated_log_ids)
    assert emd == 0.0

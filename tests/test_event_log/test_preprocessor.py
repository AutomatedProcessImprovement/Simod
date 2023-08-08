import pytest
from pix_framework.io.event_log import APROMORE_LOG_IDS, read_csv_log
from simod.event_log.preprocessor import Preprocessor


@pytest.mark.integration
@pytest.mark.parametrize("log_name", ["Simple_log_no_start_times.csv"])
def test_add_start_times(log_name, entry_point):
    log_ids = APROMORE_LOG_IDS
    event_log = read_csv_log(entry_point / log_name, log_ids)
    preprocessor = Preprocessor(event_log, log_ids)
    log = preprocessor.run()

    assert log[log_ids.start_time].isna().sum() == 0

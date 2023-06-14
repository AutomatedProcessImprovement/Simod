from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

from simod.case_attributes.discovery import discover_case_attributes


def test_discover_case_attributes(entry_point):
    log_path = entry_point / "Insurance_Claims_train.csv"
    log_ids = EventLogIDs(
        case="case_id", activity="Activity", start_time="start_time", end_time="end_time", resource="Resource"
    )
    log = read_csv_log(log_path, log_ids)

    case_attributes = discover_case_attributes(log, log_ids)

    assert len(case_attributes) > 0
    assert "extraneous_delay" in map(lambda x: x.name, case_attributes)

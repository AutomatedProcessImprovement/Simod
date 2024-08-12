from pix_framework.io.event_log import EventLogIDs, read_csv_log
from simod.data_attributes.discovery import discover_data_attributes


def test_discover_case_attributes(entry_point):
    log_path = entry_point / "Insurance_Claims_train.csv"
    log_ids = EventLogIDs(
        case="case_id", activity="activity", start_time="start_time", end_time="end_time", resource="Resource"
    )
    log = read_csv_log(log_path, log_ids)

    global_attributes, case_attributes, event_attributes = discover_data_attributes(log, log_ids)

    assert len(case_attributes) > 0
    assert "extraneous_delay" in map(lambda x: x.name, case_attributes)

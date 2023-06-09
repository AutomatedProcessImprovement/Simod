from pathlib import Path

from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

from simod.batching.discovery import discover_batching_rules

assets_dir = Path(__file__).parent / "assets"


def test_discover_batching_rules():
    log_path = assets_dir / "event_log_5.csv"
    log_ids = EventLogIDs(
        case="case_id",
        activity="Activity",
        start_time="start_time",
        end_time="end_time",
        resource="Resource",
        enabled_time="enabled_time",
        batch_id="batch_instance_id",
        batch_type="batch_instance_type",
    )
    log = read_csv_log(log_path, log_ids)

    rules = discover_batching_rules(log, log_ids)

    assert len(rules) > 0

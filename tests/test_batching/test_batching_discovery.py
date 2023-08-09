from pathlib import Path

from pix_framework.io.event_log import EventLogIDs, read_csv_log
from simod.batching.discovery import discover_batching_rules
from simod.batching.types import BatchingFiringRule

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


def test_discover_batching_rules_loanapp():
    log_path = assets_dir / "LoanApp_batch_sim_log.csv"
    log_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource",
        enabled_time="enable_time",
        batch_id="batch_instance_id",
        batch_type="batch_instance_type",
    )
    expected_rules = BatchingFiringRule(
        attribute="batch_size",
        comparison="=",
        value="3",
    )
    log = read_csv_log(log_path, log_ids)

    rules = discover_batching_rules(log, log_ids)

    assert len(rules) == 1
    assert rules[0].firing_rules[0][0] == expected_rules

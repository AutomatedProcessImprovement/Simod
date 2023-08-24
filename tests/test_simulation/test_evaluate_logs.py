from pathlib import Path

import pytest
from pix_framework.io.event_log import EventLogIDs, read_csv_log
from simod.settings.common_settings import Metric
from simod.simulation.prosimos import evaluate_logs


@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_logs(parallel):
    metrics = [
        Metric.CIRCADIAN_EMD,
        Metric.ABSOLUTE_EMD,
        Metric.CYCLE_TIME_EMD,
        Metric.TWO_GRAM_DISTANCE,
    ]

    assets_dir = Path(__file__).parent / "assets"

    log_paths = list(assets_dir.glob("*.csv"))

    log_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        resource="resource",
        start_time="start_time",
        end_time="end_time",
        enabled_time="enabled_time",
        enabling_activity="enabling_activity",
        available_time="available_time",
        estimated_start_time="estimated_start_time",
    )

    validation_log = read_csv_log(assets_dir / "validation_log.csv", log_ids)

    results = evaluate_logs(
        metrics=metrics,
        simulation_log_paths=log_paths,
        validation_log=validation_log,
        validation_log_ids=log_ids,
    )

    assert len(results) > 0

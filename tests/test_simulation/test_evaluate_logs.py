from pathlib import Path

import pytest
from pix_framework.log_ids import EventLogIDs

from simod.event_log.utilities import read
from simod.settings.common_settings import Metric
from simod.simulation.prosimos import evaluate_logs


@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_logs(parallel):
    metrics = [
        Metric.CIRCADIAN_EMD,
        Metric.ABSOLUTE_HOURLY_EMD,
        Metric.CYCLE_TIME_EMD,
        Metric.N_GRAM_DISTANCE,
    ]

    assets_dir = Path(__file__).parent / "assets"

    log_paths = list(assets_dir.glob("*.csv"))

    log_ids = EventLogIDs(
        case='case_id',
        activity='activity',
        resource='resource',
        start_time='start_time',
        end_time='end_time',
        enabled_time='enabled_time',
        enabling_activity='enabling_activity',
        available_time='available_time',
        estimated_start_time='estimated_start_time',
    )

    validation_log, _ = read(assets_dir / "validation_log.csv", log_ids)

    results = evaluate_logs(
        metrics=metrics,
        simulation_log_paths=log_paths,
        validation_log=validation_log,
        validation_log_ids=log_ids,
        run_parallel=parallel,
    )

    assert len(results) > 0

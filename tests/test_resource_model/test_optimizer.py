from pathlib import Path

import pandas as pd
import pytest
from hyperopt import STATUS_OK
from pix_framework.discovery.case_arrival import discover_case_arrival_model
from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities
from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import CalendarType
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.discovery.resource_calendar_and_performance.fuzzy.resource_calendar import FuzzyResourceCalendar
from pix_framework.discovery.resource_calendar_and_performance.resource_activity_performance import (
    ActivityResourceDistribution,
)
from pix_framework.discovery.resource_model import ResourceModel
from pix_framework.discovery.resource_profiles import ResourceProfile
from pix_framework.filesystem.file_manager import create_folder, get_random_folder_id
from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.io.event_log import APROMORE_LOG_IDS

from simod.event_log.event_log import EventLog
from simod.resource_model.optimizer import ResourceModelOptimizer
from simod.resource_model.settings import HyperoptIterationParams
from simod.settings.common_settings import Metric
from simod.settings.resource_model_settings import ResourceModelSettings
from simod.simulation.parameters.BPS_model import BPSModel

PROJECT_DIR = Path(__file__).parent.parent.parent

resource_model_config_single_values = {
    "optimization_metric": "absolute_hourly_emd",
    "num_iterations": 5,
    "resource_profiles": {
        "discovery_type": "pool",
        "granularity": [15, 60],
        "confidence": 0.05,
        "support": 0.5,
        "participation": 0.4,
    },
}

resource_model_config_intervals = {
    "optimization_metric": "circadian_emd",
    "num_iterations": 5,
    "resource_profiles": {
        "discovery_type": "differentiated",
        "granularity": [15, 60],
        "confidence": [0.05, 0.4],
        "support": [0.5, 0.8],
        "participation": [0.2, 0.6],
    },
}

resource_model_config_fuzzy = {
    "optimization_metric": "circadian_emd",
    "num_iterations": 5,
    "resource_profiles": {
        "discovery_type": "differentiated_fuzzy",
        "granularity": [15, 60],
        "fuzzy_angle": [0.2, 0.8],
    },
}

test_cases = [
    {
        "name": "Single values",
        "settings": resource_model_config_single_values,
        "event_log": "Resource_model_optimization_test.csv",
        "process_model": "Resource_model_optimization_test.bpmn",
    },
    {
        "name": "Intervals",
        "settings": resource_model_config_intervals,
        "event_log": "Resource_model_optimization_test.csv",
        "process_model": "Resource_model_optimization_test.bpmn",
    },
    {
        "name": "Fuzzy",
        "settings": resource_model_config_fuzzy,
        "event_log": "Resource_model_optimization_test.csv",
        "process_model": "Resource_model_optimization_test.bpmn",
    },
]


@pytest.mark.integration
@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["name"] for test_data in test_cases])
def test_resource_model_optimizer(entry_point, test_data):
    base_dir = PROJECT_DIR / "outputs" / get_random_folder_id(prefix="test_resource_model_optimizer_")
    create_folder(base_dir)
    log_path = entry_point / test_data["event_log"]
    event_log = EventLog.from_path(log_path, APROMORE_LOG_IDS)
    process_model_path = entry_point / test_data["process_model"]

    case_arrival_model = discover_case_arrival_model(
        event_log=event_log.train_validation_partition,
        log_ids=event_log.log_ids,
    )
    gateway_probabilities = compute_gateway_probabilities(
        event_log=event_log.train_validation_partition,
        log_ids=event_log.log_ids,
        bpmn_graph=BPMNGraph.from_bpmn_path(process_model_path),
    )
    bps_model = BPSModel(
        process_model=process_model_path,
        gateway_probabilities=gateway_probabilities,
        case_arrival_model=case_arrival_model,
    )

    settings = ResourceModelSettings.from_dict(test_data["settings"])
    optimizer = ResourceModelOptimizer(
        event_log=event_log,
        bps_model=bps_model,
        settings=settings,
        base_directory=base_dir,
    )
    result = optimizer.run()

    # Assert generic result properties and fields
    assert type(result) is HyperoptIterationParams
    assert result.process_model_path == process_model_path
    assert result.output_dir is not None
    assert result.output_dir.exists()
    # Assert discovery parameters depending on the algorithm
    if test_data["name"] == "Single values":
        assert result.optimization_metric == Metric.ABSOLUTE_EMD
        assert result.calendar_discovery_params.discovery_type == CalendarType.DIFFERENTIATED_BY_POOL
        assert (
            float(test_data["settings"]["resource_profiles"]["granularity"][0])
            <= result.calendar_discovery_params.granularity
            <= float(test_data["settings"]["resource_profiles"]["granularity"][1])
        )
        assert result.calendar_discovery_params.confidence == 0.05
        assert result.calendar_discovery_params.support == 0.5
        assert result.calendar_discovery_params.participation == 0.4
    elif test_data["name"] == "Intervals":
        assert result.optimization_metric == Metric.CIRCADIAN_EMD
        assert result.calendar_discovery_params.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE
        assert (
            float(test_data["settings"]["resource_profiles"]["granularity"][0])
            <= result.calendar_discovery_params.granularity
            <= float(test_data["settings"]["resource_profiles"]["granularity"][1])
        )
        assert (
            float(test_data["settings"]["resource_profiles"]["confidence"][0])
            <= result.calendar_discovery_params.confidence
            <= float(test_data["settings"]["resource_profiles"]["confidence"][1])
        )
        assert (
            float(test_data["settings"]["resource_profiles"]["support"][0])
            <= result.calendar_discovery_params.support
            <= float(test_data["settings"]["resource_profiles"]["support"][1])
        )
        assert (
            float(test_data["settings"]["resource_profiles"]["participation"][0])
            <= result.calendar_discovery_params.participation
            <= float(test_data["settings"]["resource_profiles"]["participation"][1])
        )
    elif test_data["name"] == "Fuzzy":
        assert result.optimization_metric == Metric.CIRCADIAN_EMD
        assert result.calendar_discovery_params.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE_FUZZY
        assert (
            float(test_data["settings"]["resource_profiles"]["granularity"][0])
            <= result.calendar_discovery_params.granularity
            <= float(test_data["settings"]["resource_profiles"]["granularity"][1])
        )
        assert (
            float(test_data["settings"]["resource_profiles"]["fuzzy_angle"][0])
            <= result.calendar_discovery_params.fuzzy_angle
            <= float(test_data["settings"]["resource_profiles"]["fuzzy_angle"][1])
        )
    else:
        assert False
    # Assert the discovered resource model contains all the elements
    assert optimizer.best_bps_model.resource_model is not None
    assert type(optimizer.best_bps_model.resource_model) == ResourceModel
    assert optimizer.best_bps_model.resource_model.resource_profiles is not None
    assert len(optimizer.best_bps_model.resource_model.resource_profiles) > 0
    assert type(optimizer.best_bps_model.resource_model.resource_profiles[0]) == ResourceProfile
    assert optimizer.best_bps_model.resource_model.resource_calendars is not None
    assert len(optimizer.best_bps_model.resource_model.resource_calendars) > 0
    assert (type(optimizer.best_bps_model.resource_model.resource_calendars[0]) == RCalendar) or (
        type(optimizer.best_bps_model.resource_model.resource_calendars[0]) == FuzzyResourceCalendar
    )
    assert optimizer.best_bps_model.resource_model.activity_resource_distributions is not None
    assert len(optimizer.best_bps_model.resource_model.activity_resource_distributions) > 0
    assert (
        type(optimizer.best_bps_model.resource_model.activity_resource_distributions[0]) == ActivityResourceDistribution
    )
    # Assert that the returned result actually has the smallest distance
    assert len(optimizer.evaluation_measurements) > 0
    iteration_results = pd.DataFrame(optimizer._bayes_trials.results).sort_values(by="loss", ascending=True)
    assert iteration_results[iteration_results["status"] == STATUS_OK].iloc[0]["output_dir"] == result.output_dir

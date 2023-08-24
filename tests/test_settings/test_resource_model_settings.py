import pytest
from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import CalendarType
from simod.settings.common_settings import Metric
from simod.settings.resource_model_settings import ResourceModelSettings

settings_single_values = {
    "optimization_metric": "absolute_hourly_emd",
    "num_iterations": 2,
    "num_evaluations_per_iteration": 3,
    "resource_profiles": {
        "discovery_type": "pool",
        "granularity": 60,
        "confidence": 0.05,
        "support": 0.5,
        "participation": 0.4,
    },
}
settings_interval_values = {
    "optimization_metric": "circadian_emd",
    "num_iterations": 2,
    "num_evaluations_per_iteration": 3,
    "resource_profiles": {
        "discovery_type": "differentiated",
        "granularity": [15, 60],
        "confidence": [0.05, 0.4],
        "support": [0.5, 0.8],
        "participation": [0.2, 0.6],
    },
}

test_cases = [
    {"name": "Single values", "resource_model": settings_single_values},
    {"name": "Intervals", "resource_model": settings_interval_values},
]


@pytest.mark.parametrize("test_data", test_cases, ids=list(map(lambda x: x["name"], test_cases)))
def test_resource_model_settings(test_data: dict):
    settings = ResourceModelSettings.from_dict(test_data["resource_model"])

    if test_data["name"] == "Single values":
        assert settings.num_iterations == settings_single_values["num_iterations"]
        assert settings.optimization_metric == Metric.ABSOLUTE_EMD
        assert settings.discovery_type == CalendarType.DIFFERENTIATED_BY_POOL
        assert settings.granularity == 60
        assert settings.confidence == 0.05
        assert settings.support == 0.5
        assert settings.participation == 0.4
    elif test_data["name"] == "Intervals":
        assert settings.num_iterations == settings_single_values["num_iterations"]
        assert settings.optimization_metric == Metric.CIRCADIAN_EMD
        assert settings.discovery_type == CalendarType.DIFFERENTIATED_BY_RESOURCE
        assert settings.granularity == (15, 60)
        assert settings.confidence == (0.05, 0.4)
        assert settings.support == (0.5, 0.8)
        assert settings.participation == (0.2, 0.6)
    else:
        assert False

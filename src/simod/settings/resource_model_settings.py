from dataclasses import dataclass
from typing import Union, Optional, Tuple

from pix_framework.discovery.resource_calendars import CalendarType

from simod.settings.common_settings import Metric
from simod.utilities import parse_single_value_or_interval


@dataclass
class ResourceModelSettings:
    """
    Resource Model optimization settings.
    """

    optimization_metric: Metric = Metric.CIRCADIAN_EMD
    max_evaluations: int = 10  # number of iterations for the optimization process
    num_evaluations_per_iteration: int = 3
    discovery_type: CalendarType = CalendarType.UNDIFFERENTIATED
    granularity: Optional[Union[int, Tuple[int, int]]] = (15, 60)  # minutes per granule
    confidence: Optional[Union[float, Tuple[float, float]]] = (0.5, 0.85)  # from 0 to 1.0
    support: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.3)  # from 0 to 1.0
    participation: Optional[Union[float, Tuple[float, float]]] = 0.4  # from 0 to 1.0
    discover_prioritization_rules: Optional[bool] = False
    discover_batching_rules: Optional[bool] = False

    @staticmethod
    def from_dict(config: dict) -> "ResourceModelSettings":
        optimization_metric = Metric.from_str(config.get("optimization_metric", "circadian_emd"))
        max_iterations = config.get("max_evaluations", 10)
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)
        discover_prioritization_rules = config.get("discover_prioritization_rules", False)
        discover_batching_rules = config.get("discover_batching_rules", False)

        resource_profiles = config.get("resource_profiles", {})
        discovery_type = CalendarType.from_str(resource_profiles.get("discovery_type", "undifferentiated"))

        # Calendar discovery parameters
        if discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL,
        ]:
            granularity = parse_single_value_or_interval(resource_profiles.get("granularity", (15, 60)))
            confidence = parse_single_value_or_interval(resource_profiles.get("confidence", (0.5, 0.85)))
            support = parse_single_value_or_interval(resource_profiles.get("support", (0.01, 0.3)))
            participation = parse_single_value_or_interval(resource_profiles.get("participation", 0.4))
        else:
            granularity, confidence, support, participation = None, None, None, None

        return ResourceModelSettings(
            optimization_metric=optimization_metric,
            max_evaluations=max_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
            discovery_type=discovery_type,
            granularity=granularity,
            confidence=confidence,
            support=support,
            participation=participation,
            discover_prioritization_rules=discover_prioritization_rules,
            discover_batching_rules=discover_batching_rules,
        )

    def to_dict(self) -> dict:
        # Parse general settings
        dictionary = {
            "optimization_metric": self.optimization_metric.value,
            "max_evaluations": self.max_evaluations,
            "num_evaluations_per_iteration": self.num_evaluations_per_iteration,
            "discovery_type": self.discovery_type.value,
            "discover_prioritization_rules": self.discover_prioritization_rules,
            "discover_batching_rules": self.discover_batching_rules,
        }

        # Parse calendar discovery parameters
        if self.discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL,
        ]:
            dictionary["granularity"] = self.granularity
            dictionary["confidence"] = self.confidence
            dictionary["support"] = self.support
            dictionary["participation"] = self.participation

        return dictionary

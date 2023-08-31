from extraneous_activity_delays.config import (
    OptimizationMetric as ExtraneousDelaysOptimizationMetric,
    DiscoveryMethod as ExtraneousDelaysDiscoveryMethod,
)
from pydantic import BaseModel

from simod.settings.common_settings import Metric


class ExtraneousDelaysSettings(BaseModel):
    optimization_metric: ExtraneousDelaysOptimizationMetric = ExtraneousDelaysOptimizationMetric.RELATIVE_EMD
    discovery_method: ExtraneousDelaysDiscoveryMethod = ExtraneousDelaysDiscoveryMethod.COMPLEX
    num_iterations: int = 1
    num_evaluations_per_iteration: int = 3

    @staticmethod
    def from_dict(config: dict) -> "ExtraneousDelaysSettings":
        optimization_metric = ExtraneousDelaysSettings._match_metric(
            config.get("optimization_metric", "relative_event_distribution")
        )
        discovery_method = ExtraneousDelaysSettings._match_method(config.get("discovery_method", "eclipse-aware"))
        num_iterations = config.get("num_iterations", 1)
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)

        return ExtraneousDelaysSettings(
            optimization_metric=optimization_metric,
            discovery_method=discovery_method,
            num_iterations=num_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
        )

    def to_dict(self) -> dict:
        return {
            "optimization_metric": str(self.optimization_metric.name),
            "discovery_method": str(self.discovery_method.name),
            "num_iterations": self.num_iterations,
            "num_evaluations_per_iteration": self.num_evaluations_per_iteration,
        }

    @staticmethod
    def _match_metric(metric: str) -> ExtraneousDelaysOptimizationMetric:
        metric = Metric.from_str(metric)
        if metric == Metric.ABSOLUTE_EMD:
            return ExtraneousDelaysOptimizationMetric.ABSOLUTE_EMD
        elif metric == Metric.CYCLE_TIME_EMD:
            return ExtraneousDelaysOptimizationMetric.CYCLE_TIME
        elif metric == Metric.CIRCADIAN_EMD:
            return ExtraneousDelaysOptimizationMetric.CIRCADIAN_EMD
        elif metric == Metric.RELATIVE_EMD:
            return ExtraneousDelaysOptimizationMetric.RELATIVE_EMD
        else:
            raise ValueError(f"Unknown extraneous delays optimization metric {metric}")

    @staticmethod
    def _match_method(method: str) -> ExtraneousDelaysDiscoveryMethod:
        if method.lower() in ["naive", "naiv", "naiiv"]:
            return ExtraneousDelaysDiscoveryMethod.NAIVE
        elif method.lower() in ["complex", "eclipse-aware", "eclipseaware", "eclipse aware"]:
            return ExtraneousDelaysDiscoveryMethod.COMPLEX
        else:
            raise ValueError(f"Unknown extraneous delays discovery method {method}")

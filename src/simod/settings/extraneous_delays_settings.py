from dataclasses import dataclass
from typing import Union

from extraneous_activity_delays.config import OptimizationMetric as ExtraneousActivityDelaysOptimizationMetric

from simod.settings.common_settings import Metric


@dataclass
class ExtraneousDelaysSettings:
    optimization_metric: ExtraneousActivityDelaysOptimizationMetric
    num_iterations: int = 1
    num_evaluations_per_iteration: int = 3

    @staticmethod
    def default() -> "ExtraneousDelaysSettings":
        return ExtraneousDelaysSettings(
            optimization_metric=ExtraneousActivityDelaysOptimizationMetric.RELATIVE_EMD,
            num_iterations=1,
            num_evaluations_per_iteration=3,
        )

    @staticmethod
    def from_dict(config: Union[dict, None]) -> Union["ExtraneousDelaysSettings", None]:
        if config is None:
            return None

        optimization_metric = ExtraneousDelaysSettings._match_metric(
            config.get("optimization_metric", "relative_event_distribution")
        )
        num_iterations = config.get("num_iterations", 1)
        num_evaluations_per_iteration = config.get("num_evaluations_per_iteration", 3)

        return ExtraneousDelaysSettings(
            optimization_metric=optimization_metric,
            num_iterations=num_iterations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
        )

    def to_dict(self) -> dict:
        return {
            "optimization_metric": str(self.optimization_metric.name),
            "num_iterations": self.num_iterations,
            "num_evaluations_per_iteration": self.num_evaluations_per_iteration,
        }

    @staticmethod
    def _match_metric(metric: str) -> ExtraneousActivityDelaysOptimizationMetric:
        metric = Metric.from_str(metric)
        if metric == Metric.ABSOLUTE_EMD:
            return ExtraneousActivityDelaysOptimizationMetric.ABSOLUTE_EMD
        elif metric == Metric.CYCLE_TIME_EMD:
            return ExtraneousActivityDelaysOptimizationMetric.CYCLE_TIME
        elif metric == Metric.CIRCADIAN_EMD:
            return ExtraneousActivityDelaysOptimizationMetric.CIRCADIAN_EMD
        elif metric == Metric.RELATIVE_EMD:
            return ExtraneousActivityDelaysOptimizationMetric.RELATIVE_EMD
        else:
            raise ValueError(f"Unknown metric {metric}")

from dataclasses import dataclass
from typing import Union

from extraneous_activity_delays.config import OptimizationMetric as ExtraneousActivityDelaysOptimizationMetric


@dataclass
class ExtraneousDelaysSettings:
    optimization_metric: ExtraneousActivityDelaysOptimizationMetric
    num_iterations: int = 10

    @staticmethod
    def default() -> "ExtraneousDelaysSettings":
        return ExtraneousDelaysSettings(
            optimization_metric=ExtraneousActivityDelaysOptimizationMetric.RELATIVE_EMD,
            num_iterations=10,
        )

    @staticmethod
    def from_dict(config: Union[dict, None]) -> Union["ExtraneousDelaysSettings", None]:
        if config is None:
            return None

        optimization_metric = config.get("optimization_metric")
        if optimization_metric is not None:
            optimization_metric = ExtraneousDelaysSettings._match_metric(optimization_metric)
        else:
            optimization_metric = ExtraneousActivityDelaysOptimizationMetric.RELATIVE_EMD

        num_iterations = config.get("num_iterations", 10)

        return ExtraneousDelaysSettings(
            optimization_metric=optimization_metric,
            num_iterations=num_iterations,
        )

    def to_dict(self) -> dict:
        return {
            "optimization_metric": str(self.optimization_metric.name),
            "num_iterations": self.num_iterations,
        }

    @staticmethod
    def _match_metric(metric: str) -> ExtraneousActivityDelaysOptimizationMetric:
        metric = metric.lower()

        if metric == "absolute_emd":
            return ExtraneousActivityDelaysOptimizationMetric.ABSOLUTE_EMD
        elif metric == "cycle_time":
            return ExtraneousActivityDelaysOptimizationMetric.CYCLE_TIME
        elif metric == "circadian_emd":
            return ExtraneousActivityDelaysOptimizationMetric.CIRCADIAN_EMD
        elif metric == "relative_emd":
            return ExtraneousActivityDelaysOptimizationMetric.RELATIVE_EMD
        else:
            raise ValueError(f"Unknown metric {metric}")

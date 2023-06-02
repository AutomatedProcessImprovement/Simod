from dataclasses import dataclass

from pix_framework.statistics.distribution import DurationDistribution


@dataclass
class ExtraneousDelay:
    activity_name: str
    duration_distribution: DurationDistribution

    def to_dict(self) -> dict:
        return {
            "activity": self.activity_name,
            "duration_distribution": self.duration_distribution.to_prosimos_distribution(),
        }

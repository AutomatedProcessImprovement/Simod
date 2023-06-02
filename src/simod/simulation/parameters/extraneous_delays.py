from dataclasses import dataclass
from typing import List, Dict

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


def convert_extraneous_delays_to_extraneous_package_format(
    delays: List[ExtraneousDelay],
) -> Dict[str, DurationDistribution]:
    return {delay.activity_name: delay.duration_distribution for delay in delays}

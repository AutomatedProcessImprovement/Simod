from dataclasses import dataclass

from pix_framework.statistics.distribution import DurationDistribution


@dataclass
class ExtraneousDelay:
    activity_name: str
    delay_id: str
    duration_distribution: DurationDistribution

    def to_dict(self) -> dict:
        return {
            "activity": self.activity_name,
            "event_id": self.delay_id,
        } | self.duration_distribution.to_prosimos_distribution()

    @staticmethod
    def from_dict(delay: dict) -> "ExtraneousDelay":
        return ExtraneousDelay(
            activity_name=delay["activity"],
            delay_id=delay["event_id"],
            duration_distribution=DurationDistribution.from_dict(delay),
        )

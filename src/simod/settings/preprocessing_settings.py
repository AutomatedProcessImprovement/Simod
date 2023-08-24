from pix_framework.enhancement.start_time_estimator.config import ConcurrencyThresholds
from pydantic import BaseModel


class PreprocessingSettings(BaseModel):
    multitasking: bool = False
    enable_time_concurrency_threshold: float = 0.5
    concurrency_thresholds: ConcurrencyThresholds = ConcurrencyThresholds(df=0.75, l2l=0.9, l1l=0.9)

    @staticmethod
    def from_dict(config: dict) -> "PreprocessingSettings":
        return PreprocessingSettings(
            multitasking=config.get("multitasking", False),
            enable_time_concurrency_threshold=config.get("enable_time_concurrency_threshold", 0.5),
            concurrency_thresholds=ConcurrencyThresholds(
                df=config.get("concurrency_df", 0.9),
                l2l=config.get("concurrency_l2l", 0.9),
                l1l=config.get("concurrency_l1l", 0.9),
            ),
        )

    def to_dict(self) -> dict:
        return {
            "multitasking": self.multitasking,
            "enable_time_concurrency_threshold": self.enable_time_concurrency_threshold,
            "concurrency_df": self.concurrency_thresholds.df,
            "concurrency_l2l": self.concurrency_thresholds.l2l,
            "concurrency_l1l": self.concurrency_thresholds.l1l,
        }

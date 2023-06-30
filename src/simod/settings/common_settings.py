from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, List, Optional

from pix_framework.log_ids import EventLogIDs, DEFAULT_XES_IDS

from ..utilities import get_project_dir

QBP_NAMESPACE_URI = "http://www.qbp-simulator.com/Schema201212"
BPMN_NAMESPACE_URI = "http://www.omg.org/spec/BPMN/20100524/MODEL"
PROJECT_DIR = get_project_dir()


class Metric(str, Enum):
    DL = "dl"
    CIRCADIAN_EMD = "circadian_emd"
    ABSOLUTE_HOURLY_EMD = "absolute_hourly_emd"
    CYCLE_TIME_EMD = "cycle_time_emd"
    N_GRAM_DISTANCE = "n_gram_distance"

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> "Union[Metric, List[Metric]]":
        if isinstance(value, str):
            return Metric._from_str(value)
        elif isinstance(value, list):
            return [Metric._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> "Metric":
        if value.lower() == "dl":
            return cls.DL
        elif value.lower() == "n_gram_distance":
            return cls.N_GRAM_DISTANCE
        elif value.lower() == "circadian_emd":
            return cls.CIRCADIAN_EMD
        elif value.lower() in ("absolute_hourly_emd", "absolute_hour_emd", "abs_hourly_emd", "abs_hour_emd"):
            return cls.ABSOLUTE_HOURLY_EMD
        elif value.lower() == "cycle_time_emd":
            return cls.CYCLE_TIME_EMD
        else:
            raise ValueError(f"Unknown value {value}")

    def __str__(self):
        if self == Metric.DL:
            return "DL"
        elif self == Metric.N_GRAM_DISTANCE:
            return "N_GRAM_DISTANCE"
        elif self == Metric.CIRCADIAN_EMD:
            return "CIRCADIAN_EMD"
        elif self == Metric.ABSOLUTE_HOURLY_EMD:
            return "ABSOLUTE_HOURLY_EMD"
        elif self == Metric.CYCLE_TIME_EMD:
            return "CYCLE_TIME_EMD"
        return f"Unknown Metric {str(self)}"


@dataclass
class CommonSettings:
    log_path: Path
    test_log_path: Optional[Path]
    log_ids: Optional[EventLogIDs]
    model_path: Optional[Path]
    repetitions: int
    evaluation_metrics: Union[Metric, List[Metric]]
    clean_intermediate_files: bool = True

    @staticmethod
    def default() -> "CommonSettings":
        return CommonSettings(
            log_path=Path("example_log.csv"),
            test_log_path=None,
            log_ids=DEFAULT_XES_IDS,
            model_path=None,
            repetitions=1,
            evaluation_metrics=[
                Metric.DL,
                Metric.N_GRAM_DISTANCE,
                Metric.ABSOLUTE_HOURLY_EMD,
                Metric.CIRCADIAN_EMD,
                Metric.CYCLE_TIME_EMD,
            ],
            clean_intermediate_files=True,
        )

    @staticmethod
    def from_dict(config: dict) -> "CommonSettings":
        log_path = Path(config["log_path"])
        if not log_path.is_absolute():
            log_path = PROJECT_DIR / log_path

        test_log_path = config.get("test_log_path", None)
        if test_log_path is not None:
            test_log_path = Path(test_log_path)
            if not test_log_path.is_absolute():
                test_log_path = PROJECT_DIR / test_log_path

        metrics = [Metric.from_str(metric) for metric in config["evaluation_metrics"]]

        mapping = config.get("log_ids", None)
        if mapping is not None:
            log_ids = EventLogIDs.from_dict(mapping)
        else:
            log_ids = DEFAULT_XES_IDS

        clean_up = config.get("clean_intermediate_files", True)

        model_path = config.get("model_path", None)
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_DIR / model_path

        return CommonSettings(
            log_path=log_path,
            test_log_path=test_log_path,
            log_ids=log_ids,
            model_path=model_path,
            repetitions=config["repetitions"],
            evaluation_metrics=metrics,
            clean_intermediate_files=clean_up,
        )

    def to_dict(self) -> dict:
        return {
            "log_path": str(self.log_path),
            "test_log_path": str(self.test_log_path) if self.test_log_path is not None else None,
            "log_ids": self.log_ids.to_dict(),
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "repetitions": self.repetitions,
            "evaluation_metrics": [str(metric) for metric in self.evaluation_metrics],
            "clean_intermediate_files": self.clean_intermediate_files,
        }

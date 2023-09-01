from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs
from pydantic import BaseModel

from ..utilities import get_project_dir

QBP_NAMESPACE_URI = "http://www.qbp-simulator.com/Schema201212"
BPMN_NAMESPACE_URI = "http://www.omg.org/spec/BPMN/20100524/MODEL"
PROJECT_DIR = get_project_dir()


class Metric(str, Enum):
    DL = "dl"
    TWO_GRAM_DISTANCE = "two_gram_distance"
    THREE_GRAM_DISTANCE = "three_gram_distance"
    CIRCADIAN_EMD = "circadian_event_distribution"
    ARRIVAL_EMD = "arrival_event_distribution"
    RELATIVE_EMD = "relative_event_distribution"
    ABSOLUTE_EMD = "absolute_event_distribution"
    CYCLE_TIME_EMD = "cycle_time_distribution"

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
        elif value.lower() in ["two_gram_distance", "2_gram_distance"]:
            return cls.TWO_GRAM_DISTANCE
        elif value.lower() in ["n_gram", "n_gram_distance", "three_gram_distance", "3_gram_distance"]:
            return cls.THREE_GRAM_DISTANCE
        elif value.lower() in ["circadian_event_distribution", "circadian_emd"]:
            return cls.CIRCADIAN_EMD
        elif value.lower() in ["arrival_event_distribution", "arrival_emd"]:
            return cls.ARRIVAL_EMD
        elif value.lower() in ["relative_event_distribution", "relative_emd"]:
            return cls.RELATIVE_EMD
        elif value.lower() in [
            "absolute_event_distribution",
            "absolute_hourly_emd",
            "absolute_hour_emd",
            "abs_hourly_emd",
            "abs_hour_emd",
        ]:
            return cls.ABSOLUTE_EMD
        elif value.lower() in ["cycle_time_distribution", "cycle_time_emd"]:
            return cls.CYCLE_TIME_EMD
        else:
            raise ValueError(f"Unknown value {value}")

    def __str__(self):
        if self == Metric.DL:
            return "DL"
        elif self == Metric.TWO_GRAM_DISTANCE:
            return "TWO_GRAM_DISTANCE"
        elif self == Metric.THREE_GRAM_DISTANCE:
            return "THREE_GRAM_DISTANCE"
        elif self == Metric.CIRCADIAN_EMD:
            return "CIRCADIAN_EVENT_DISTRIBUTION"
        elif self == Metric.ARRIVAL_EMD:
            return "ARRIVAL_EVENT_DISTRIBUTION"
        elif self == Metric.RELATIVE_EMD:
            return "RELATIVE_EVENT_DISTRIBUTION"
        elif self == Metric.ABSOLUTE_EMD:
            return "ABSOLUTE_EVENT_DISTRIBUTION"
        elif self == Metric.CYCLE_TIME_EMD:
            return "CYCLE_TIME_DISTRIBUTION"
        return f"Unknown Metric {str(self)}"


class CommonSettings(BaseModel):
    # Log & Model parameters
    train_log_path: Path = Path("default_path.csv")
    log_ids: EventLogIDs = PROSIMOS_LOG_IDS
    test_log_path: Optional[Path] = None
    process_model_path: Optional[Path] = None
    # Final evaluation parameters
    perform_final_evaluation: bool = False
    num_final_evaluations: int = 10
    evaluation_metrics: List[Metric] = field(default_factory=list)
    # Common config
    use_observed_arrival_distribution: bool = False
    clean_intermediate_files: bool = True
    discover_case_attributes: bool = False

    @staticmethod
    def from_dict(config: dict, config_dir: Optional[Path] = None) -> "CommonSettings":
        base_files_dir = config_dir or Path.cwd()

        # Training log path
        train_log_path = Path(config["train_log_path"])
        if not train_log_path.is_absolute():
            train_log_path = base_files_dir / train_log_path

        # Log IDs
        if "log_ids" in config:
            log_ids = EventLogIDs.from_dict(config["log_ids"])
        else:
            log_ids = PROSIMOS_LOG_IDS

        # Test log path
        if "test_log_path" in config and config["test_log_path"] is not None:
            test_log_path = Path(config["test_log_path"])
            if not test_log_path.is_absolute():
                test_log_path = base_files_dir / test_log_path
        else:
            test_log_path = None

        # Process model path
        if "process_model_path" in config and config["process_model_path"] is not None:
            process_model_path = Path(config["process_model_path"])
            if not process_model_path.is_absolute():
                process_model_path = base_files_dir / process_model_path
        else:
            process_model_path = None

        # Flag to perform final evaluation (set to true if there is a test log)
        if test_log_path is not None:
            perform_final_evaluation = True
        else:
            perform_final_evaluation = config.get("perform_final_evaluation", False)

        # Number of final evaluations & metrics to evaluate
        if perform_final_evaluation:
            num_final_evaluations = config.get("num_final_evaluations", 10)
            if "evaluation_metrics" in config:
                metrics = [Metric.from_str(metric) for metric in config["evaluation_metrics"]]
            else:
                metrics = [
                    Metric.DL,
                    Metric.TWO_GRAM_DISTANCE,
                    Metric.THREE_GRAM_DISTANCE,
                    Metric.CIRCADIAN_EMD,
                    Metric.ARRIVAL_EMD,
                    Metric.RELATIVE_EMD,
                    Metric.ABSOLUTE_EMD,
                    Metric.CYCLE_TIME_EMD,
                ]
        else:
            num_final_evaluations = 0
            metrics = []

        # Quality check
        if perform_final_evaluation and num_final_evaluations == 0:
            print(
                "Wrong configuration! perform_final_evaluation=True but "
                "num_final_evaluations=0. Setting to 10 by default."
            )
            num_final_evaluations = 10

        use_observed_arrival_distribution = config.get("use_observed_arrival_distribution", False)
        clean_up = config.get("clean_intermediate_files", True)
        discover_case_attributes = config.get("discover_case_attributes", False)

        return CommonSettings(
            train_log_path=train_log_path,
            log_ids=log_ids,
            test_log_path=test_log_path,
            process_model_path=process_model_path,
            perform_final_evaluation=perform_final_evaluation,
            num_final_evaluations=num_final_evaluations,
            evaluation_metrics=metrics,
            use_observed_arrival_distribution=use_observed_arrival_distribution,
            clean_intermediate_files=clean_up,
            discover_case_attributes=discover_case_attributes,
        )

    def to_dict(self) -> dict:
        return {
            "train_log_path": str(self.train_log_path),
            "test_log_path": str(self.test_log_path) if self.test_log_path is not None else None,
            "log_ids": self.log_ids.to_dict(),
            "process_model_path": str(self.process_model_path) if self.process_model_path is not None else None,
            "num_final_evaluations": self.num_final_evaluations,
            "evaluation_metrics": [str(metric) for metric in self.evaluation_metrics],
            "use_observed_arrival_distribution": self.use_observed_arrival_distribution,
            "clean_intermediate_files": self.clean_intermediate_files,
            "discover_case_attributes": self.discover_case_attributes,
        }

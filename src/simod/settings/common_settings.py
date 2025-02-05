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
    """
    Enum class storing the metrics used to evaluate the quality of a BPS model.

    Attributes
    ----------
    DL : str
        Control-flow Log Distance metric based in the Damerau-Levenshtein distance.
    TWO_GRAM_DISTANCE : str
        Two-gram distance metric.
    THREE_GRAM_DISTANCE : str
        Three-gram distance metric.
    CIRCADIAN_EMD : str
        Earth Mover's Distance (EMD) for circadian event distribution.
    CIRCADIAN_WORKFORCE_EMD : str
        EMD for circadian workforce distribution.
    ARRIVAL_EMD : str
        EMD for arrival event distribution.
    RELATIVE_EMD : str
        EMD for relative event distribution.
    ABSOLUTE_EMD : str
        EMD for absolute event distribution.
    CYCLE_TIME_EMD : str
        EMD for cycle time distribution.
    """

    DL = "dl"
    TWO_GRAM_DISTANCE = "two_gram_distance"
    THREE_GRAM_DISTANCE = "three_gram_distance"
    CIRCADIAN_EMD = "circadian_event_distribution"
    CIRCADIAN_WORKFORCE_EMD = "circadian_workforce_distribution"
    ARRIVAL_EMD = "arrival_event_distribution"
    RELATIVE_EMD = "relative_event_distribution"
    ABSOLUTE_EMD = "absolute_event_distribution"
    CYCLE_TIME_EMD = "cycle_time_distribution"

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> "Union[Metric, List[Metric]]":
        """
        Converts a string (or list of strings) representing metric names into an instance (or list of instances)
        of the :class:`Metric` enum.

        Parameters
        ----------
        value : Union[str, List[str]]
            A string representing a metric name or a list of metric names.

        Returns
        -------
        Union[:class:`Metric`, List[:class:`Metric`]]
            An instance of :class:`Metric` if a single string is provided,
            or a list of :class:`Metric` instances if a list of strings is provided.

        Raises
        ------
        ValueError
            If the provided string does not match any metric name.
        """
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
        elif value.lower() in ["circadian_workforce_distribution", "workforce_emd", "workforce_distribution"]:
            return cls.CIRCADIAN_WORKFORCE_EMD
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
        elif self == Metric.CIRCADIAN_WORKFORCE_EMD:
            return "CIRCADIAN_WORKFORCE_DISTRIBUTION"
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
    """
    General configuration parameters of SIMOD and parameters common to all pipeline stages

    Attributes
    ----------
        train_log_path : :class:`~pathlib.Path`
            Path to the training log (the one used to discover the BPS model).
        log_ids : :class:`EventLogIDs`
            Dataclass storing the mapping between the column names in the CSV and their role (case_id, activity, etc.).
        test_log_path : :class:`~pathlib.Path`, optional
            Path to the event log to perform the final evaluation of the discovered BPS model (if desired).
        process_model_path : :class:`~pathlib.Path`, optional
            Path to the BPMN model for the control-flow (skip its discovery and use this one).
        perform_final_evaluation : bool
            Boolean indicating whether to perform the final evaluation of the discovered BPS model.
            If true, either use the event log in [test_log_path] if specified, or split the training log to obtain a
            testing set.
        num_final_evaluations : int
            Number of replications of the final evaluation to perform.
        evaluation_metrics : list
            List of :class:`Metric` evaluation metrics to use in the final evaluation.
        use_observed_arrival_distribution : bool
            Boolean indicating whether to use the distribution of observed case arrival times (true), or to discover a
            probability distribution function to model them (false).
        clean_intermediate_files : bool
            Boolean indicating whether to delete all intermediate created files.
        discover_data_attributes : bool
            Boolean indicating whether to discover data attributes and their creation/update rules.

    """
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
    discover_data_attributes: bool = False

    @staticmethod
    def from_dict(config: dict, config_dir: Optional[Path] = None) -> "CommonSettings":
        """
        Instantiates the SIMOD common configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with the configuration values for the SIMOD common parameters.
        config_dir : :class:`~pathlib.Path`, optional
            If the path to the event log(s) is specified in a relative manner, ``[config_dir]`` is used to complete
            such paths. If ``None``, relative paths are complemented with the current directory.

        Returns
        -------
        :class:`CommonSettings`
            Instance of the SIMOD common configuration for the specified dictionary values.
        """
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
                    Metric.CIRCADIAN_WORKFORCE_EMD,
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
        discover_data_attributes = config.get("discover_data_attributes", False)

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
            discover_data_attributes=discover_data_attributes,
        )

    def to_dict(self) -> dict:
        """
        Translate the common configuration stored in this instance into a dictionary.

        Returns
        -------
        dict
            Python dictionary storing this configuration.
        """
        return {
            "train_log_path": str(self.train_log_path),
            "test_log_path": str(self.test_log_path) if self.test_log_path is not None else None,
            "log_ids": self.log_ids.to_dict(),
            "process_model_path": str(self.process_model_path) if self.process_model_path is not None else None,
            "num_final_evaluations": self.num_final_evaluations,
            "evaluation_metrics": [str(metric) for metric in self.evaluation_metrics],
            "use_observed_arrival_distribution": self.use_observed_arrival_distribution,
            "clean_intermediate_files": self.clean_intermediate_files,
            "discover_data_attributes": self.discover_data_attributes,
        }

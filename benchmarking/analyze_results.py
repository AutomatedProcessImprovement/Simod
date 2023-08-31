from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

simod_version = "3.6.0"
results_dir = Path(__file__).parent / Path(f"results/{simod_version}/diff_observed-arrivals")

metric_names_mapping = {
    "absolute_event_distribution": "AED",
    "arrival_event_distribution": "CAR",
    "circadian_event_distribution": "CED",
    "cycle_time_distribution": "CTD",
    "relative_event_distribution": "RED",
    "three_gram_distance": "NGD(3)",
    "two_gram_distance": "NGD",
}

event_log_names_mapping = {
    "BPIC_2012_train": "BPIC12",
    "BPIC_2017_train": "BPIC17",
    "CallCenter_train": "CALL",
    "AcademicCredentials_train": "AC_CRE",
}


@dataclass
class DiscoveryResult:
    result_dir: Path

    _evaluation_measures_path: Optional[Path] = None
    _evaluation_measures: Optional[pd.DataFrame] = None
    _simulated_log_paths: Optional[list[Path]] = None
    _name: Optional[str] = None

    def __post_init__(self):
        evaluation_dir = self.result_dir / "evaluation"
        self._evaluation_measures_path = next(evaluation_dir.glob("evaluation_*.csv"))
        self._simulated_log_paths = list((evaluation_dir / "simulation").glob("simulated_*.csv"))
        self._name = next(self.result_dir.glob("*.bpmn")).stem
        self._name = event_log_names_mapping[self._name]
        self._evaluation_measures = pd.read_csv(self._evaluation_measures_path).drop(columns=["run_num"])
        self._evaluation_measures["name"] = self._name
        self._rename_column_values("metric", metric_names_mapping)

    def _rename_column_values(self, column_name: str, mapping: dict[str, str]):
        self._evaluation_measures[column_name] = self._evaluation_measures[column_name].apply(
            lambda item: mapping[item]
        )

    @property
    def evaluation_measures(self) -> pd.DataFrame:
        return self._evaluation_measures

    @property
    def mean_evaluation_measures(self) -> pd.DataFrame:
        return self.evaluation_measures.groupby(["metric"]).mean(numeric_only=True).assign(name=self.name).reset_index()

    @property
    def name(self) -> str:
        return self._name


# Current measurements
results = [DiscoveryResult(result_dir / "best_result") for result_dir in results_dir.iterdir() if result_dir.is_dir()]
mean_evaluation_measures = pd.concat([result.mean_evaluation_measures for result in results]).reset_index(drop=True)
mean_evaluation_measures["simod_version"] = simod_version

# Save measurements
mean_evaluation_measures.to_csv("measurements.csv", index=False)

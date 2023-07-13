from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

results_dir = Path("results")

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
    "BPIC_2012_train.csv": "BPIC12",
    "BPIC_2017_train.csv": "BPIC17",
    "CallCenter_train.csv": "CALL",
    "AcademicCredentials_train.csv": "AC_CRE",
}


@dataclass
class DiscoveryResult:
    result_dir: Path

    _evaluation_measures_path: Optional[Path] = None
    _evaluation_measures: Optional[pd.DataFrame] = None
    _simulated_log_paths: Optional[list[Path]] = None
    _name: Optional[str] = None

    def __post_init__(self):
        self._evaluation_measures_path = next(self.result_dir.glob("evaluation_*.csv"))
        self._simulated_log_paths = list((self.result_dir / "simulation").glob("simulated_*.csv"))
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
mean_evaluation_measures["simod_version"] = "3.5.24"

# Previous measurements
previous_results = pd.DataFrame(
    [
        {
            "name": "AC_CRE",
            "metric": "AED",
            "distance": 117.32,
        },
        {
            "name": "AC_CRE",
            "metric": "CAR",
            "distance": 110.38,
        },
        {
            "name": "AC_CRE",
            "metric": "CED",
            "distance": 3.11,
        },
        {
            "name": "AC_CRE",
            "metric": "RED",
            "distance": 48.19,
        },
        {
            "name": "AC_CRE",
            "metric": "CTD",
            "distance": 62.23,
        },
        {
            "name": "AC_CRE",
            "metric": "NGD",
            "distance": 0.24,
        },
        {
            "name": "BPIC12",
            "metric": "AED",
            "distance": 313.30,
        },
        {
            "name": "BPIC12",
            "metric": "CAR",
            "distance": 336.42,
        },
        {
            "name": "BPIC12",
            "metric": "CED",
            "distance": 2.10,
        },
        {
            "name": "BPIC12",
            "metric": "RED",
            "distance": 96.82,
        },
        {
            "name": "BPIC12",
            "metric": "CTD",
            "distance": 93.45,
        },
        {
            "name": "BPIC12",
            "metric": "NGD",
            "distance": 0.56,
        },
        {
            "name": "BPIC17",
            "metric": "AED",
            "distance": 314.92,
        },
        {
            "name": "BPIC17",
            "metric": "CAR",
            "distance": 390.04,
        },
        {
            "name": "BPIC17",
            "metric": "CED",
            "distance": 1.65,
        },
        {
            "name": "BPIC17",
            "metric": "RED",
            "distance": 132.31,
        },
        {
            "name": "BPIC17",
            "metric": "CTD",
            "distance": 102.85,
        },
        {
            "name": "BPIC17",
            "metric": "NGD",
            "distance": 0.37,
        },
        {
            "name": "CALL",
            "metric": "AED",
            "distance": 61.76,
        },
        {
            "name": "CALL",
            "metric": "CAR",
            "distance": 61.68,
        },
        {
            "name": "CALL",
            "metric": "CED",
            "distance": 4.72,
        },
        {
            "name": "CALL",
            "metric": "RED",
            "distance": 0.0,
        },
        {
            "name": "CALL",
            "metric": "CTD",
            "distance": 8.18,
        },
        {
            "name": "CALL",
            "metric": "NGD",
            "distance": 0.08,
        },
    ]
)
previous_results["simod_version"] = "2023.03"

# Both measurements
both_results = pd.concat([mean_evaluation_measures, previous_results]).reset_index(drop=True)
both_results.to_csv("results.csv", index=False)

# Comparison
comparison = both_results.pivot_table(
    index=["name", "metric"],
    columns=["simod_version"],
    values=["distance"],
    aggfunc="mean",
).reset_index()

comparison.columns = ["_".join(col).strip() for col in comparison.columns.values]
comparison = comparison.rename(columns={"name_": "name", "metric_": "metric"})
comparison["distance_diff"] = comparison["distance_3.5.24"] - comparison["distance_2023.03"]
comparison.to_csv("comparison.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=comparison, x="metric", y="distance_diff", hue="name", ax=ax)
ax.set_xlabel("Metric")
ax.set_ylabel("Distance difference")
ax.set_title("Distance difference between Simod 3.5.24 and 2023.03")
plt.tight_layout()
plt.savefig("comparison.png")

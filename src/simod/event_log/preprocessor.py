from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.enhancement.multitasking import adjust_durations
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyThresholds
from pix_framework.enhancement.start_time_estimator.config import Configuration as StartTimeEstimatorConfiguration
from pix_framework.enhancement.start_time_estimator.estimator import StartTimeEstimator
from pix_framework.io.event_log import EventLogIDs

from simod.cli_formatter import print_section, print_step


@dataclass
class MultitaskingSettings:
    log_path: Path
    output_dir: Path
    is_concurrent: bool
    verbose: bool


@dataclass
class Settings:
    multitasking_settings: Optional[MultitaskingSettings] = None


class Preprocessor:
    """
    Preprocessor executes event log pre-processing according to the `run()` arguments and returns the modified log back.
    """

    _log: pd.DataFrame
    _log_ids: EventLogIDs

    def __init__(self, log: pd.DataFrame, log_ids: EventLogIDs):
        keys = [log_ids.start_time, log_ids.end_time] if log_ids.start_time in log.columns else [log_ids.end_time]
        self._log = log.sort_values(by=keys).reset_index(drop=True)
        self._log_ids = log_ids

    def run(
        self,
        multitasking: bool = False,
        concurrency_thresholds: ConcurrencyThresholds = ConcurrencyThresholds(),
        enable_time_concurrency_threshold: float = 0.75,
    ) -> pd.DataFrame:
        """
        Executes all pre-processing steps and updates the configuration if necessary.

        Start times discovery is always executed if the log does not contain the start time column.

        :param multitasking: Whether to adjust the timestamps for multitasking.
        :param concurrency_thresholds: Thresholds for the Heuristics Miner to estimate start/enabled times.
        :param enable_time_concurrency_threshold: Threshold for the Heuristics Miner to estimate enabled times.
        :return: The pre-processed event log.
        """
        print_section("Pre-processing")

        if self._log_ids.start_time not in self._log.columns or self._log[self._log_ids.start_time].isnull().any():
            self._add_start_times(concurrency_thresholds)

        if multitasking:
            self._adjust_for_multitasking()

        if self._log_ids.enabled_time not in self._log.columns:
            # The start times were not estimated (otherwise enabled times would
            # be present), and the enabled times are not in the original log
            self._add_enabled_times(enable_time_concurrency_threshold)

        return self._log

    def _adjust_for_multitasking(self, verbose=False):
        print_step("Adjusting timestamps for multitasking")

        self._log = adjust_durations(
            self._log,
            self._log_ids,
            verbose=verbose,
        )

    def _add_start_times(self, concurrency_thresholds: ConcurrencyThresholds):
        print_step("Adding start times")

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self._log_ids,
            concurrency_thresholds=concurrency_thresholds,
        )

        self._log = StartTimeEstimator(self._log, configuration).estimate(replace_recorded_start_times=True)

    def _add_enabled_times(self, enable_time_concurrency_threshold: float):
        print_step("Adding enabled times")

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self._log_ids,
            concurrency_thresholds=ConcurrencyThresholds(df=enable_time_concurrency_threshold),
            consider_start_times=True,
        )
        # The start times are the original ones, so use overlapping concurrency oracle
        OverlappingConcurrencyOracle(self._log, configuration).add_enabled_times(self._log)

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typing import Optional

from estimate_start_times.concurrency_oracle import OverlappingConcurrencyOracle
from estimate_start_times.config import Configuration as StartTimeEstimatorConfiguration, HeuristicsThresholds
from estimate_start_times.estimator import StartTimeEstimator
from simod.cli_formatter import print_step, print_section
from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.multitasking import adjust_durations


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

    def run(self, multitasking: bool = False, concurrency_thresholds: HeuristicsThresholds = HeuristicsThresholds()) -> pd.DataFrame:
        """
        Executes all pre-processing steps and updates the configuration if necessary.

        Start times discovery is always executed if the log does not contain the start time column.

        :param multitasking: Whether to adjust the timestamps for multitasking.
        :param concurrency_thresholds: Thresholds for the Heuristics Miner to estimate start/enabled times.
        :return: The pre-processed event log.
        """
        print_section('Pre-processing')

        if self._log_ids.start_time not in self._log.columns:
            self._add_start_times(concurrency_thresholds)

        if multitasking:
            self._adjust_for_multitasking()

        if self._log_ids.enabled_time not in self._log.columns:
            # The start times were not estimated (otherwise enabled times would
            # be present), and the enabled times are not in the original log
            self._add_enabled_times(concurrency_thresholds)

        return self._log

    def _adjust_for_multitasking(self, is_concurrent=False, verbose=False):
        print_step('Adjusting timestamps for multitasking')

        self._log = adjust_durations(
            self._log,
            self._log_ids,
            output_path=None,
            is_concurrent=is_concurrent,
            verbose=verbose,
        )

    def _add_start_times(self, concurrency_thresholds: HeuristicsThresholds):
        print_step('Adding start times')

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self._log_ids,
            heuristics_thresholds=concurrency_thresholds,
        )

        self._log = StartTimeEstimator(
            self._log,
            configuration
        ).estimate(
            replace_recorded_start_times=True
        )

    def _add_enabled_times(self, concurrency_thresholds: HeuristicsThresholds):
        print_step('Adding enabled times')

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self._log_ids,
            heuristics_thresholds=concurrency_thresholds,
            consider_start_times=True,
        )
        # The start times are the original ones, so use overlapping concurrency oracle
        OverlappingConcurrencyOracle(self._log, configuration).add_enabled_times(self._log)

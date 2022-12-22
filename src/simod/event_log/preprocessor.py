from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from estimate_start_times.config import Configuration as StartTimeEstimatorConfiguration
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
        self._log = log.copy()
        self._log_ids = log_ids

    def run(self, multitasking: bool = False) -> pd.DataFrame:
        """
        Executes all pre-processing steps and updates the configuration if necessary.

        Start times discovery is always executed if the log does not contain the start time column.

        :param multitasking: Whether to adjust the timestamps for multitasking.
        :return: The pre-processed event log.
        """
        print_section('Pre-processing')

        if self._log_ids.start_time not in self._log.columns:
            self._add_start_times()

        if multitasking:
            self._adjust_for_multitasking()

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

    def _add_start_times(self):
        print_step('Adding start times')

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self._log_ids,
        )

        self._log = StartTimeEstimator(self._log, configuration).estimate(replace_recorded_start_times=True)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from estimate_start_times.config import Configuration as StartTimeEstimatorConfiguration
from estimate_start_times.estimator import StartTimeEstimator
from simod.cli_formatter import print_step, print_section, print_notice
from simod.configuration import Configuration
from simod.event_log.event_log import EventLog
from simod.event_log.multitasking import adjust_durations
from simod.event_log.utilities import read
from simod.utilities import remove_asset


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
    Preprocessor executes any event log pre-processing required according to the configuration.
    """

    config: Configuration
    output_dir: Path
    log: Optional[pd.DataFrame]

    _csv_log_path: Optional[Path]
    _tmp_dirs: [Path] = []

    def __init__(self, config: Configuration, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

        self.log, self._csv_log_path = read(self.config.common.log_path, self.config.common.log_ids)

    def run(self) -> Tuple[Configuration, EventLog]:
        """
        Executes all pre-processing steps and updates the configuration if necessary.
        """
        print_section('Pre-processing')

        if self.config.common.log_ids.start_time not in self.log.columns:
            self._add_start_times()

        if self.config.preprocessing.multitasking is True:
            self._adjust_for_multitasking()

        event_log = EventLog.from_df(
            self.log,
            self.config.common.log_ids,
            log_path=self.config.common.log_path,
            csv_log_path=self._csv_log_path,
        )

        return self.config, event_log

    def _adjust_for_multitasking(self, is_concurrent=False, verbose=False):
        print_step('Adjusting timestamps for multitasking')

        processed_log_path = self.output_dir / (self.config.common.log_path.stem + '_processed.xes')

        self.log = adjust_durations(self.log, self.config.common.log_ids, processed_log_path,
                                    is_concurrent=is_concurrent, verbose=verbose)
        self.config.log_path = processed_log_path
        self._tmp_dirs.append(processed_log_path)
        print_notice(f'New log path: {self.config.log_path}')

    def _add_start_times(self):
        print_step('Adding start times')

        configuration = StartTimeEstimatorConfiguration(
            log_ids=self.config.common.log_ids,
        )

        assert self.log is not None, 'Log is None'

        extended_event_log = StartTimeEstimator(self.log, configuration).estimate(replace_recorded_start_times=True)
        self.log = extended_event_log

    def cleanup(self):
        for folder in self._tmp_dirs:
            remove_asset(folder)

from pathlib import Path
from typing import Optional

import pandas as pd

from simod.cli_formatter import print_step, print_section, print_notice
from simod.event_log.multitasking import adjust_durations
from simod.event_log.utilities import read
from simod.optimization.settings import OptimizationSettings
from simod.utilities import remove_asset


class Preprocessor:
    """Preprocessor executes any event log pre-processing required according to the configuration."""
    config: OptimizationSettings
    log: Optional[pd.DataFrame] = None

    _tmp_dirs: [Path] = []

    def __init__(self, config: OptimizationSettings):
        self.config = config

    def _multitasking_processing(self, log_path: Path, output_dir: Path, is_concurrent=False, verbose=False):
        print_step('Multitasking pre-processing')
        self.log, log_path_csv = read(log_path)
        processed_log_path = output_dir / (log_path.stem + '_processed.xes')
        self.log = adjust_durations(self.log, processed_log_path, is_concurrent=is_concurrent, verbose=verbose)
        self.config.log_path = processed_log_path
        self._tmp_dirs.append(processed_log_path)
        print_notice(f'New log path: {self.config.log_path}')

    def run(self) -> OptimizationSettings:
        """Executes all pre-processing steps and updates the configuration if necessary."""
        print_section('Pre-processing')

        if self.config.adjust_for_multitasking:
            self._multitasking_processing(self.config.project_settings.log_path,
                                          self.config.project_settings.output_dir)

        return self.config

    def cleanup(self):
        for folder in self._tmp_dirs:
            remove_asset(folder)

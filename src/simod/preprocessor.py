from pathlib import Path

from .cli_formatter import print_step, print_section, print_notice
from .configuration import Configuration
from .multitasking import adjust_durations


class Preprocessor:
    """Preprocessor executes any pre-processing required according to the configuration."""
    config: Configuration

    def __init__(self, config: Configuration):
        self.config = config

    def _multitasking_processing(self, log_path: Path, output_dir: Path, is_concurrent=False, verbose=False):
        print_step('Multitasking pre-processing')
        processed_log_path = output_dir / (log_path.stem + '_processed.xes')
        adjust_durations(log_path, processed_log_path, is_concurrent=is_concurrent, verbose=verbose)
        self.config.log_path = processed_log_path
        print_notice(f'New log path: {self.config.log_path}')

    def run(self) -> Configuration:
        """run function executes all pre-processing steps and updates the configuration if necessary."""
        print_section('Pre-processing')

        if self.config.multitasking:
            self._multitasking_processing(self.config.log_path, self.config.output)

        return self.config

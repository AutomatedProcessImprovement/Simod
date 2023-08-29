from pathlib import Path
from typing import Optional

import pandas as pd
from pix_framework.io.event_log import DEFAULT_XES_IDS, EventLogIDs, read_csv_log
from pix_framework.io.event_log import split_log_training_validation_trace_wise as split_log

from .preprocessor import Preprocessor
from .utilities import convert_df_to_xes
from ..settings.preprocessing_settings import PreprocessingSettings
from ..utilities import get_process_name_from_log_path


class EventLog:
    """
    Event log class that contains the log and its splits, and column names.

    Use static methods to create an EventLog from a path in other ways that are implemented.
    """

    train_partition: pd.DataFrame
    validation_partition: pd.DataFrame
    train_validation_partition: pd.DataFrame
    test_partition: pd.DataFrame
    log_ids: EventLogIDs
    process_name: str  # a name of the process that is used mainly for file names

    def __init__(
        self,
        log_train: pd.DataFrame,
        log_validation: pd.DataFrame,
        log_train_validation: pd.DataFrame,
        log_test: pd.DataFrame,
        log_ids: EventLogIDs,
        process_name: Optional[str] = None,
    ):
        self.train_partition = log_train
        self.validation_partition = log_validation
        self.train_validation_partition = log_train_validation
        self.test_partition = log_test
        self.log_ids = log_ids

        if process_name is not None:
            self.process_name = process_name
        else:
            self.process_name = "business_process"

    @staticmethod
    def from_path(
        train_log_path: Path,
        log_ids: EventLogIDs,
        preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
        need_test_partition: Optional[bool] = False,
        process_name: Optional[str] = None,
        test_log_path: Optional[Path] = None,
        split_ratio: float = 0.8,
    ) -> "EventLog":
        """
        Loads an event log from a file and does the log split for training, validation, and test.
        """
        # Check event log prerequisites
        if not train_log_path.name.endswith(".csv") and not train_log_path.name.endswith(".csv.gz"):
            raise ValueError(
                f"The specified training log has an unsupported extension ({train_log_path.name}). "
                f"Only 'csv' and 'csv.gz' supported."
            )
        if test_log_path is not None:
            if not test_log_path.name.endswith(".csv") and not test_log_path.name.endswith(".csv.gz"):
                raise ValueError(
                    f"The specified test log has an unsupported extension ({test_log_path.name}). "
                    f"Only 'csv' and 'csv.gz' supported."
                )

        # Read training event log
        event_log = read_csv_log(train_log_path, log_ids)

        # Preprocess training event log
        preprocessor = Preprocessor(event_log, log_ids)
        processed_event_log = preprocessor.run(
            multitasking=preprocessing_settings.multitasking,
            enable_time_concurrency_threshold=preprocessing_settings.enable_time_concurrency_threshold,
            concurrency_thresholds=preprocessing_settings.concurrency_thresholds,
        )

        # Get test if needed, and split train+validation
        if test_log_path is not None:
            # Test log provided, the input log is train+validation
            train_validation_df = processed_event_log
            test_df = read_csv_log(test_log_path, log_ids)
        elif need_test_partition:
            # Test log not provided but needed, split input into test and train+validation
            train_validation_df, test_df = split_log(processed_event_log, log_ids, training_percentage=split_ratio)
        else:
            # Test log not provided and not needed, the input log is train+validation
            train_validation_df = processed_event_log
            test_df = None
        train_df, validation_df = split_log(train_validation_df, log_ids, training_percentage=split_ratio)

        # Return EventLog instance with different partitions
        return EventLog(
            log_train=train_df,
            log_validation=validation_df,
            log_train_validation=train_validation_df,
            log_test=test_df,
            log_ids=log_ids,
            process_name=get_process_name_from_log_path(train_log_path) if process_name is None else process_name,
        )

    def train_to_xes(self, path: Path):
        """
        Saves the training log to a XES file.
        """
        write_xes(self.train_partition, self.log_ids, path)

    def validation_to_xes(self, path: Path):
        """
        Saves the validation log to a XES file.
        """
        write_xes(self.validation_partition, self.log_ids, path)

    def train_validation_to_xes(self, path: Path):
        """
        Saves the validation log to a XES file.
        """
        write_xes(self.train_validation_partition, self.log_ids, path)

    def test_to_xes(self, path: Path):
        """
        Saves the test log to a XES file.
        """
        write_xes(self.test_partition, self.log_ids, path)


def write_xes(
    log: pd.DataFrame,
    log_ids: EventLogIDs,
    output_path: Path,
):
    """
    Writes the log to a file in XES format.
    """
    df = log.rename(
        columns={
            log_ids.activity: "concept:name",
            log_ids.case: "case:concept:name",
            log_ids.resource: "org:resource",
            log_ids.start_time: "start_timestamp",
            log_ids.end_time: "time:timestamp",
        }
    )

    df = df[
        [
            "case:concept:name",
            "concept:name",
            "org:resource",
            "start_timestamp",
            "time:timestamp",
        ]
    ]

    df.fillna("UNDEFINED", inplace=True)

    convert_df_to_xes(df, DEFAULT_XES_IDS, output_path)

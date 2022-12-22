from pathlib import Path
from typing import Optional

import pandas as pd

from extraneous_activity_delays.utils.log_split import split_log_training_validation_event_wise as split_log
from .column_mapping import EventLogIDs, STANDARD_COLUMNS
from .utilities import read, convert_df_to_xes


class EventLog:
    """
    Event log class that contains the log and its splits, and column names.

    Use static methods to create an EventLog from a path in other ways that are implemented.
    """

    process_name: str  # a name of the process that is used mainly for file names
    log_ids: EventLogIDs
    log_path: Optional[Path]
    log_csv_path: Optional[Path]  # XES is converted to CSV and stored by Simod
    train_partition: pd.DataFrame
    validation_partition: pd.DataFrame
    test_partition: pd.DataFrame

    def __init__(
            self,
            log_ids: EventLogIDs,
            log_train: pd.DataFrame,
            log_validation: pd.DataFrame,
            log_test: pd.DataFrame,
            process_name: Optional[str] = None,
            log_path: Optional[Path] = None,
            csv_log_path: Optional[Path] = None,
    ):
        self.log_ids = log_ids
        self.log_path = log_path
        self.log_csv_path = csv_log_path
        self.train_partition = log_train
        self.validation_partition = log_validation
        self.test_partition = log_test

        if process_name is not None:
            self.process_name = process_name
        elif log_path is not None:
            self.process_name = log_path.stem
        else:
            self.process_name = 'business_process'

    @staticmethod
    def from_df(
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            test_log: Optional[pd.DataFrame] = None,
            process_name: Optional[str] = None,
            log_path: Optional[Path] = None,
            csv_log_path: Optional[Path] = None,
    ):
        """
        Creates an EventLog from a DataFrame.
        """
        split_ratio = 0.8

        if test_log is None:
            train_validation_df, test_df = split_log(log, log_ids, training_percentage=split_ratio)
            train_df, validation_df = split_log(train_validation_df, log_ids, training_percentage=split_ratio)
        else:
            train_df, validation_df = split_log(log, log_ids, training_percentage=split_ratio)
            test_df = test_log

        return EventLog(
            log_ids=log_ids,
            log_train=train_df,
            log_validation=validation_df,
            log_test=test_df,
            process_name=process_name,
            log_path=log_path,
            csv_log_path=csv_log_path,
        )

    @staticmethod
    def from_path(
            path: Path,
            log_ids: EventLogIDs,
            process_name: Optional[str] = None,
            test_path: Optional[Path] = None,
    ) -> 'EventLog':
        """
        Loads an event log from a file and does the log split for training, validation, and test.
        """
        split_ratio = 0.8

        df, csv_path = read(path, log_ids)

        if test_path is None:
            train_validation_df, test_df = split_log(df, log_ids, training_percentage=split_ratio)
            train_df, validation_df = split_log(train_validation_df, log_ids, training_percentage=split_ratio)
        else:
            train_df, validation_df = split_log(df, log_ids, training_percentage=split_ratio)
            test_df, _ = read(test_path, log_ids)

        return EventLog(
            log_ids=log_ids,
            log_path=path,
            csv_log_path=csv_path,
            log_train=train_df.sort_values(by=log_ids.start_time),
            log_validation=validation_df.sort_values(by=log_ids.start_time),
            log_test=test_df.sort_values(by=log_ids.start_time),
            process_name=process_name,
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
    df = log.rename(columns={
        log_ids.activity: 'concept:name',
        log_ids.case: 'case:concept:name',
        log_ids.resource: 'org:resource',
        log_ids.start_time: 'start_timestamp',
        log_ids.end_time: 'time:timestamp',
    })

    df = df[[
        'case:concept:name',
        'concept:name',
        'org:resource',
        'start_timestamp',
        'time:timestamp',
    ]]

    df.fillna('UNDEFINED', inplace=True)

    convert_df_to_xes(df, STANDARD_COLUMNS, output_path)

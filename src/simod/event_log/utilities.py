import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pendulum
from openxes_cli.lib import xes_to_csv, csv_to_xes
from pix_framework.log_ids import DEFAULT_XES_IDS, EventLogIDs
from start_time_estimator.config import Configuration as StartTimeEstimatorConfiguration
from start_time_estimator.estimator import StartTimeEstimator


def convert_xes_to_csv_if_needed(log_path: Path, output_path: Optional[Path] = None) -> Path:
    _, ext = os.path.splitext(log_path)
    if ext == ".xes":
        if output_path:
            log_path_csv = output_path
        else:
            log_path_csv = log_path.with_suffix(".csv")
        convert_xes_to_csv(log_path, log_path_csv)
        return log_path_csv
    else:
        return log_path


def read(log_path: Path, log_ids: EventLogIDs = DEFAULT_XES_IDS) -> Tuple[pd.DataFrame, Path]:
    """Reads an event log from XES or CSV and converts timestamp to UTC.

    :param log_path: Path to the event log.
    :param log_ids: Column names of the event log.
    :return: A tuple containing the event log dataframe and the path to CSV file.
    """
    log_path_csv = convert_xes_to_csv_if_needed(log_path)
    log = pd.read_csv(log_path_csv)
    convert_timestamps(log, log_ids)
    log[log_ids.resource].fillna("NOT_SET", inplace=True)
    log[log_ids.resource] = log[log_ids.resource].astype(str)
    return log, log_path_csv


def convert_timestamps(log: pd.DataFrame, log_ids: EventLogIDs):
    time_columns = [
        log_ids.start_time,
        log_ids.end_time,
        log_ids.enabled_time,
        log_ids.available_time,
        log_ids.estimated_start_time,
    ]
    for name in time_columns:
        if name in log.columns:
            log[name] = pd.to_datetime(log[name], utc=True, format="ISO8601")


def convert_xes_to_csv(xes_path: Path, csv_path: Path):
    return xes_to_csv(xes_path, csv_path)


def convert_df_to_xes(df: pd.DataFrame, log_ids: EventLogIDs, output_path: Path):
    xes_datetime_format = "YYYY-MM-DDTHH:mm:ss.SSSZ"
    df[log_ids.start_time] = df[log_ids.start_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format)
    )
    df[log_ids.end_time] = df[log_ids.end_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format)
    )
    df.to_csv(output_path, index=False)
    csv_to_xes(output_path, output_path)


def add_enabled_time_if_missing(log: pd.DataFrame, log_ids: EventLogIDs) -> pd.DataFrame:
    if log_ids.enabled_time in log.columns:
        return log

    configuration = StartTimeEstimatorConfiguration(log_ids=log_ids)
    log = StartTimeEstimator(log, configuration).estimate()

    return log

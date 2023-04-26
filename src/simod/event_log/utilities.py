import os
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree

import pandas as pd
import pendulum

from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS


def convert_xes_to_csv_if_needed(log_path: Path, output_path: Optional[Path] = None) -> Path:
    _, ext = os.path.splitext(log_path)
    if ext == '.xes':
        if output_path:
            log_path_csv = output_path
        else:
            log_path_csv = log_path.with_suffix('.csv')
        convert_xes_to_csv(log_path, log_path_csv)
        return log_path_csv
    else:
        return log_path


def read(log_path: Path, log_ids: EventLogIDs = STANDARD_COLUMNS) -> Tuple[pd.DataFrame, Path]:
    """Reads an event log from XES or CSV and converts timestamp to UTC.

    :param log_path: Path to the event log.
    :param log_ids: Column names of the event log.
    :return: A tuple containing the event log dataframe and the path to CSV file.
    """
    log_path_csv = convert_xes_to_csv_if_needed(log_path)
    log = pd.read_csv(log_path_csv)
    convert_timestamps(log, log_ids)
    log[log_ids.resource].fillna('NOT_SET', inplace=True)
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
            log[name] = pd.to_datetime(log[name], utc=True)


def convert_xes_to_csv(xes_path: Path, csv_path: Path):
    # Prepare args
    args = ['poetry', 'run', 'pm4py_wrapper', '-i', str(xes_path), '-o', str(csv_path.parent), 'xes-to-csv']
    # Run command
    os.system(' '.join(args))


def convert_df_to_xes(df: pd.DataFrame, log_ids: EventLogIDs, output_path: Path):
    # Format timestamp events
    xes_datetime_format = 'YYYY-MM-DDTHH:mm:ss.SSSZ'
    df[log_ids.start_time] = df[log_ids.start_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    df[log_ids.end_time] = df[log_ids.end_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    # Export CSV file
    csv_path = Path(str(output_path).replace(".xes", ".csv"))
    df.to_csv(csv_path, index=False)
    # Prepare args
    args = ["poetry", "run", "pm4py_wrapper", "-i", "\"" + str(csv_path) + "\"", "-o", "\"" + str(output_path.parent) + "\"", "csv-to-xes"]
    # Run command
    os.system(' '.join(args))
    # Remove tmp CSV file
    csv_path.unlink()


def reformat_timestamps(xes_path: Path, output_path: Path):
    """Converts timestamps in XES to a format suitable for the Simod's calendar Java dependency."""
    ns = 'http://www.xes-standard.org/'
    date_tag = f'{{{ns}}}date'

    ElementTree.register_namespace('', ns)
    tree = ElementTree.parse(xes_path)
    root = tree.getroot()
    xpaths = [
        ".//*[@key='time:timestamp']",
        ".//*[@key='start_timestamp']"
    ]
    for xpath in xpaths:
        for element in root.iterfind(xpath):
            try:
                timestamp = pd.to_datetime(element.get('value'), format='%Y-%m-%d %H:%M:%S')
                value = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')
                element.set('value', value)
                element.tag = date_tag
            except ValueError:
                continue
    tree.write(output_path)

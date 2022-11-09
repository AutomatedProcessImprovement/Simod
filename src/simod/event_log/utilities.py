import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import pendulum

from simod.cli_formatter import print_step
from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS


def remove_outliers(event_log: pd.DataFrame, log_ids: EventLogIDs) -> pd.DataFrame:  # TODO: must be tested
    if not isinstance(event_log, pd.DataFrame):
        raise TypeError('Event log must be a pandas DataFrame')

    case_duration_key = 'duration_seconds'

    # calculating case durations
    cases_durations = list()
    for case_id, trace in event_log.groupby(log_ids.case):
        duration = (trace[log_ids.end_time].max() - trace[log_ids.start_time].min()).total_seconds()
        cases_durations.append({log_ids.case: case_id, case_duration_key: duration})
    cases_durations = pd.DataFrame(cases_durations)

    # merging data
    event_log = event_log.merge(cases_durations, how='left', on=log_ids.case)

    # filtering rare events
    unique_cases_durations = event_log[[log_ids.case, case_duration_key]].drop_duplicates()
    first_quantile = unique_cases_durations.quantile(0.1)
    last_quantile = unique_cases_durations.quantile(0.9)
    event_log = event_log[(event_log[case_duration_key] <= last_quantile.duration_seconds) & (
            event_log.duration_seconds >= first_quantile.duration_seconds)]
    event_log = event_log.drop(columns=[case_duration_key])

    return event_log


def convert_xes_to_csv_if_needed(log_path: Path, output_path: Optional[Path] = None) -> Path:
    _, ext = os.path.splitext(log_path)
    if ext != '.csv':
        if output_path:
            log_path_csv = output_path
        else:
            log_path_csv = log_path.with_suffix('.csv')
        convert_xes_to_csv(log_path, log_path_csv)
        return log_path_csv
    else:
        return log_path


def read(log_path: Path, log_ids: EventLogIDs = STANDARD_COLUMNS) -> Tuple[pd.DataFrame, Path]:
    log_path_csv = convert_xes_to_csv_if_needed(log_path)
    log = pd.read_csv(log_path_csv)
    convert_timestamps(log, log_ids)
    return log, log_path_csv


def convert_timestamps(log: pd.DataFrame, log_ids: EventLogIDs):
    time_columns = [log_ids.start_time, log_ids.enabled_time, log_ids.end_time]
    for name in time_columns:
        if name in log.columns:
            log[name] = pd.to_datetime(log[name], utc=True)


def convert_xes_to_csv(xes_path: Path, csv_path: Path):
    args = ['pm4py_wrapper', '-i', str(xes_path), '-o', str(csv_path.parent), 'xes-to-csv']
    print_step(f'Executing shell command [convert_xes_to_csv]: {args}')
    os.system(' '.join(args))


def convert_df_to_xes(df: pd.DataFrame, log_ids: EventLogIDs, output_path: Path):
    xes_datetime_format = 'YYYY-MM-DDTHH:mm:ss.SSSZ'
    df[log_ids.start_time] = df[log_ids.start_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    df[log_ids.end_time] = df[log_ids.end_time].apply(
        lambda x: pendulum.parse(x.isoformat()).format(xes_datetime_format))
    df.to_csv(output_path, index=False)
    args = ['pm4py_wrapper', '-i', str(output_path), '-o', str(output_path.parent), 'csv-to-xes']
    print_step(f'Executing shell command [convert_df_to_xes]: {args}')
    os.system(' '.join(args))


def reformat_timestamps(xes_path: Path, output_path: Path):
    """Converts timestamps in XES to a format suitable for the Simod's calendar Java dependency."""
    ns = 'http://www.xes-standard.org/'
    date_tag = f'{{{ns}}}date'

    ET.register_namespace('', ns)
    tree = ET.parse(xes_path)
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


def sample_log(log: pd.DataFrame, log_ids: EventLogIDs):
    def sample_size(population_size, confidence_level, confidence_interval):
        confidence_level_constant = {50: .67, 68: .99, 90: 1.64, 95: 1.96, 99: 2.57}

        p = 0.5
        e = confidence_interval / 100.0
        N = population_size

        # deviations for the confidence level
        Z = confidence_level_constant[confidence_level]

        # sample size
        n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)

        # adjusting sample size for finite population
        n = n_0 / (1 + ((n_0 - 1) / float(N)))

        sample_size = int(math.ceil(n))

        return sample_size

    case_ids = list(log[log_ids.case].unique())
    if len(case_ids) > 1000:
        sample_size = sample_size(len(case_ids), 95.0, 3.0)
        sample_case_ids = random.sample(case_ids, sample_size)
        log = log[log[log_ids.case].isin(sample_case_ids)]

    return log

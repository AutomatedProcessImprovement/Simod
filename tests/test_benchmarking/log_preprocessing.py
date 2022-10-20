from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import pendulum


def process_confidential_1000():
    log_path = Path('logs/confidential_1000.csv')
    fill_missing_values(log_path, ['org:resource', 'resourceId'])


def process_confidential_2000():
    log_path = Path('logs/confidential_2000.csv')
    fill_missing_values(log_path, ['org:resource', 'resourceId'])


def fill_missing_values(log_path: Path, columns: list):
    df = pd.read_csv(log_path)

    for column in columns:
        df[column].fillna('NotSpecified', inplace=True)

    convert_timestamps(df, ['time:timestamp', 'start_timestamp'])

    df.to_csv(log_path.with_stem(log_path.stem + '_processed'), index=False)


def convert_timestamps(df: pd.DataFrame, columns: list):
    for column in columns:
        df[column] = df[column].apply(lambda x: pendulum.parse(x).to_iso8601_string())


def modify_timestamp_in_xml(log_path: Path):
    # fixing the timestamp XML tag after pm4py conversion
    ElementTree.register_namespace('', 'http://www.xes-standard.org/')
    root = ElementTree.parse(str(log_path))
    for elem in root.findall(".//*[@key='time:timestamp']"):
        elem.tag = 'date'
        elem.text = pendulum.parse(elem.text).to_datetime_string()
    root.write(str(log_path), encoding='utf-8', xml_declaration=True)


# df = pd.read_csv('logs/confidential_1000.csv')

process_confidential_1000()
process_confidential_2000()

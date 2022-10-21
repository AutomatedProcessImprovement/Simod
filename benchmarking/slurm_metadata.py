import re
from pathlib import Path

import yaml


def extract_log_data(file_path: Path):
    with file_path.open('r') as f:
        name = None
        status = None
        error_line = None
        folder = None

        for line in f:
            n = get_log_name(line)
            name = n if n is not None else name

            s = get_error_status(line)
            status = 'failed' if s is not None else status
            error_line = line.strip() if s is not None else error_line

            s = get_success_status(line)
            status = 'finished' if s is not None else status

            dir = get_folder(line)
            folder = dir if dir is not None else folder

    return {'name': name, 'status': status, 'error': error_line, 'folder': folder}


def get_log_name(log: str):
    result = re.search(r'(?<=/logs/)(.*)((?=.xes\')|(?=.xes\s))', log)
    if result:
        return result.group(1)
    return None


def get_success_status(log: str):
    result = re.search(r'Exporting canonical model', log)
    if result:
        return result.group(0)
    return None


def get_error_status(log: str):
    result = re.search(r'(error)', log, re.I)
    if result:
        return result.group(0)
    return None


def get_folder(log: str):
    result = re.search(r'(?<=outputs/)(.+?)(?=/)', log)
    if result:
        return result.group(1)
    return None


data = [
    extract_log_data(file)
    for file in Path('.').glob('slurm-*.out')
]

with open('slurm_metadata.yml', 'w') as f:
    yaml.dump(data, f)

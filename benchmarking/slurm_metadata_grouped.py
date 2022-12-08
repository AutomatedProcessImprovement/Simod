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
    result = re.search(r'(error|AllTrialsFailed)', log, re.I)
    if result:
        return result.group(0)
    return None


def get_folder(log: str):
    result = re.search(r'(?<=outputs/)(.+?)(?=/)', log)
    if result:
        return result.group(1)
    return None


# differentiated_slurm_log_names = [
#     'slurm-34258544.out',
#     'slurm-34256706.out',
#     'slurm-34258543.out',
#     'slurm-34258545.out',
#     'slurm-34258548.out',
#     'slurm-34258542.out',
#     'slurm-34258546.out',
#     'slurm-34258547.out',
#     'slurm-34254742.out',
# ]

undifferentiated_slurm_log_names = [
    'slurm-34797943.out',
    'slurm-34797944.out',
    'slurm-34797945.out',
    'slurm-34797946.out',
    'slurm-34797947.out',
    'slurm-34797948.out',
    'slurm-34797949.out',
]


def save_metadata(log_names: list, file_name: str):
    data = [
        extract_log_data(file)
        for file in Path('.').glob('slurm-*.out')
        if file.name in log_names
    ]

    with open(file_name, 'w') as f:
        yaml.dump(data, f)


# save_metadata(differentiated_slurm_log_names, 'differentiated_slurm_metadata.yml')
save_metadata(undifferentiated_slurm_log_names, 'undifferentiated_slurm_metadata.yml')

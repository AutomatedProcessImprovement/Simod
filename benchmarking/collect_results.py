import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def extract_log_data(file_path: Path):
    with file_path.open('r') as f:
        slurm_file = str(file_path)
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

    return {'slurm_file': slurm_file, 'name': name, 'status': status, 'error': error_line, 'folder': folder}


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


def copy_files(files: Iterable, dst: Path):
    for file in files:
        shutil.copy(file, dst / file.name)


def main():
    output_dir = Path('results/' + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = Path('configs')
    jobs_dir = Path('jobs')
    logs_dir = Path('logs')

    shutil.copytree(configs_dir, output_dir / 'configs')
    shutil.copytree(jobs_dir, output_dir / 'jobs')
    shutil.copytree(logs_dir, output_dir / 'logs')

    trials_dir = output_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)

    data = [
        extract_log_data(file)
        for file in Path('.').glob('slurm-*.out')
    ]

    for item in data:
        if item['status'] == 'finished':
            # Determine optimization metric for calendars

            trial_calendars_output_dir = next((Path('../outputs/') / item['folder']).glob('calendars_*'))
            calendar_evaluation_file = next(trial_calendars_output_dir.glob('evaluation_*.csv'))
            df = pd.read_csv(calendar_evaluation_file)
            metric_name = df['metric'].iloc[0]

            # Determine resource profile discovery type

            trial_canonical_model_path = Path('../outputs/') / item['folder'] / 'best_result' / 'canonical_model.json'
            with trial_canonical_model_path.open('r') as f:
                data = json.load(f)
                discovery_type = data['calendars']['resource_profiles']['discovery_type']

            # Update folder name

            item_dir = trials_dir / (metric_name + '_' + discovery_type + '_' + item['folder'])
            item_dir.mkdir(parents=True, exist_ok=True)

            # Copy files

            shutil.copy(calendar_evaluation_file, item_dir / ('calendar_' + calendar_evaluation_file.name))

            slurm_file = Path(item['slurm_file'])
            shutil.copy(slurm_file, item_dir / slurm_file.name)

            trial_best_output_dir = Path('../outputs/') / item['folder'] / 'best_result'

            model_files = trial_best_output_dir.glob('*.bpmn')
            json_files = trial_best_output_dir.glob('*.json')
            evaluation_files = trial_best_output_dir.glob('evaluation_*.csv')

            copy_files(model_files, item_dir)
            copy_files(json_files, item_dir)
            copy_files(evaluation_files, item_dir)

            simulated_log = next((trial_best_output_dir / 'simulation').glob('simulated_log_*.csv'))
            shutil.copy(simulated_log, item_dir / simulated_log.name)

        else:
            item_dir = trials_dir / item['folder']
            item_dir.mkdir(parents=True, exist_ok=True)

            slurm_file = Path(item['slurm_file'])
            shutil.copy(slurm_file, item_dir / slurm_file.name)


if __name__ == '__main__':
    main()

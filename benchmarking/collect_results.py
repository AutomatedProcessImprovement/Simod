import json
import re
import shutil
from dataclasses import dataclass
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


@dataclass
class EventLogIDs:
    case = 'case_id'
    activity = 'activity'
    enabled_time = 'enable_time'
    start_time = 'start_time'
    end_time = 'end_time'
    resource = 'resource'


prosimos_log_ids = EventLogIDs()


def get_avg_cycle_time_for_log(log_path: Path) -> float:
    global prosimos_log_ids

    df = pd.read_csv(log_path)

    df[prosimos_log_ids.start_time] = pd.to_datetime(df[prosimos_log_ids.start_time])
    df[prosimos_log_ids.end_time] = pd.to_datetime(df[prosimos_log_ids.end_time])

    cycle_times = [
        (case[prosimos_log_ids.end_time].max() - case[prosimos_log_ids.start_time].min()).total_seconds()
        for (case_id, case) in df.groupby(prosimos_log_ids.case)
    ]

    avg_cycle_time = sum(cycle_times) / len(cycle_times)

    return avg_cycle_time


def get_avg_cycle_time_for_logs(log_paths: Iterable[Path]) -> float:
    cycle_times = [get_avg_cycle_time_for_log(log_path) for log_path in log_paths]

    avg_cycle_time = sum(cycle_times) / len(cycle_times)

    return avg_cycle_time


def save_avg_cycle_time_for_test_logs(logs_dir: Path, output_dir: Path):
    log_paths = list(logs_dir.glob('*_test.csv'))

    cycle_times = [get_avg_cycle_time_for_log(log) for log in log_paths]

    df = pd.DataFrame({'cycle_time': cycle_times}, index=[log.with_suffix('').name for log in log_paths])

    df.to_csv(output_dir / 'test_cycle_times.csv')


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

    experiments = [
        extract_log_data(file)
        for file in Path('.').glob('slurm-*.out')
    ]

    metrics = ['absolute_hourly_emd', 'circadian_emd', 'cycle_time_emd']
    resource_discovery_types = ['undifferentiated', 'differentiated_by_pool', 'differentiated_by_resource']
    permutations = [f'{metric}-{resource_discovery_type}' for metric in metrics for resource_discovery_type in
                    resource_discovery_types]
    event_log_names = [
        'Production_train',
        'poc_processmining_train',
        'Governmental_Agency_train',
        'ConsultaDataMining201618_train',
        'BPIC_2017_W_contained_train',
        'BPIC_2012_W_contained_train',
    ]
    cycle_time_df = pd.DataFrame(columns=permutations, index=event_log_names)

    for experiment in experiments:
        if experiment['status'] == 'finished':
            # Determine optimization metric for calendars

            trial_calendars_output_dir = next((Path('../outputs/') / experiment['folder']).glob('calendars_*'))
            calendar_evaluation_file = next(trial_calendars_output_dir.glob('evaluation_*.csv'))
            df = pd.read_csv(calendar_evaluation_file)
            metric_name = df['metric'].iloc[0]

            # Determine resource profile discovery type

            trial_canonical_model_path = Path('../outputs/') / experiment[
                'folder'] / 'best_result' / 'canonical_model.json'
            with trial_canonical_model_path.open('r') as f:
                data = json.load(f)
                discovery_type = data['calendars']['resource_profiles']['discovery_type']

            # Update folder name

            item_dir = trials_dir / (metric_name + '_' + discovery_type + '_' + experiment['folder'])
            item_dir.mkdir(parents=True, exist_ok=True)

            # Copy files

            shutil.copy(calendar_evaluation_file, item_dir / ('calendar_' + calendar_evaluation_file.name))

            slurm_file = Path(experiment['slurm_file'])
            shutil.copy(slurm_file, item_dir / slurm_file.name)

            trial_best_output_dir = Path('../outputs/') / experiment['folder'] / 'best_result'

            model_files = list(trial_best_output_dir.glob('*.bpmn'))
            json_files = trial_best_output_dir.glob('*.json')
            evaluation_files = trial_best_output_dir.glob('evaluation_*.csv')

            copy_files(model_files, item_dir)
            copy_files(json_files, item_dir)
            copy_files(evaluation_files, item_dir)

            simulation_logs_dir = trial_best_output_dir / 'simulation'
            simulated_log_paths = list(simulation_logs_dir.glob('simulated_log_*.csv'))

            # copy folder
            # shutil.copytree(simulation_logs_dir, item_dir / simulation_logs_dir.name)

            # copy only one simulation log
            simulated_log = simulated_log_paths[0]
            shutil.copy(simulated_log, item_dir / simulated_log.name)

            # Calculate average cycle time

            dataset_name = model_files[0].with_suffix('').name
            column_name = '-'.join([metric_name, discovery_type]).lower()

            avg_cycle_time = get_avg_cycle_time_for_logs(simulated_log_paths)

            cycle_time_df.loc[dataset_name, column_name] = avg_cycle_time
        else:
            item_dir = trials_dir / experiment['folder']
            item_dir.mkdir(parents=True, exist_ok=True)

            slurm_file = Path(experiment['slurm_file'])
            shutil.copy(slurm_file, item_dir / slurm_file.name)

    cycle_time_df.to_csv(output_dir / 'cycle_times.csv')


if __name__ == '__main__':
    # main()

    logs_dir = Path('results/2022-12-08_202655/logs')
    output_dir = Path('results/2022-12-08_202655')
    save_avg_cycle_time_for_test_logs(logs_dir, output_dir)

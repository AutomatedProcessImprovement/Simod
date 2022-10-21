from pathlib import Path

import pandas as pd

import slurm_metadata


def collect_log_metrics(folder_name: str, log_name: str):
    folder = '../outputs' / Path(folder_name) / 'best_result'
    df = pd.DataFrame()
    for file in folder.glob('evaluation_*.csv'):
        df = pd.concat([df, pd.read_csv(file)])
        df['folder'] = folder.absolute()
        df['log_name'] = log_name
    return df


data = [
    slurm_metadata.extract_log_data(file)
    for file in Path('.').glob('slurm-*.out')
]

all_logs_df = pd.DataFrame()

for item in data:
    if item['status'] == 'finished':
        log_df = collect_log_metrics(item['folder'], item['name'])
        all_logs_df = pd.concat([all_logs_df, log_df])

all_logs_df.to_csv('log_metrics.csv', index=False)

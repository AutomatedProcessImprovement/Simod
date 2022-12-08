from pathlib import Path

import pandas as pd

import slurm_metadata_grouped


def collect_log_metrics(folder_name: str, log_name: str):
    folder = '../outputs' / Path(folder_name) / 'best_result'
    df = pd.DataFrame()
    for file in folder.glob('evaluation_*.csv'):
        df = pd.concat([df, pd.read_csv(file)])
        df['folder'] = folder.absolute()
        df['log_name'] = log_name
    return df


def save_metrics(log_names: list, file_name: str):
    data = [
        slurm_metadata_grouped.extract_log_data(file)
        for file in Path('.').glob('slurm-*.out')
        if file.name in log_names
    ]

    all_logs_df = pd.DataFrame()

    for item in data:
        if item['status'] == 'finished':
            log_df = collect_log_metrics(item['folder'], item['name'])
            all_logs_df = pd.concat([all_logs_df, log_df])

    all_logs_df.to_csv(file_name, index=False)


# differentiated_slurm_log_names = slurm_metadata_grouped.differentiated_slurm_log_names
undifferentiated_slurm_log_names = slurm_metadata_grouped.undifferentiated_slurm_log_names

# save_metrics(differentiated_slurm_log_names, 'differentiated_log_metrics.csv')
save_metrics(undifferentiated_slurm_log_names, 'trial 1/undifferentiated_log_metrics.csv')

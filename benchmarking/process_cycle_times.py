from pathlib import Path

import pandas as pd


# df1 = pd.read_csv('results/cycle_times.csv', index_col=0)
#
# df2 = pd.read_csv('results/test_cycle_times.csv', index_col=0)
# df2.rename(columns={'cycle_time': 'original'}, inplace=True)
#
# df = pd.concat([df1, df2], axis=1)
# df.to_csv('results/cycle_times.csv')


# df = pd.read_csv('results/cycle_times.csv', index_col=0)
#
# # find column with the closest value to 'original'
# for col in df.columns:
#     df[col] = df[col] / df['original']


def correct_log(path: Path, output_dir: Path):
    df = pd.read_csv(path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['Resource'] = df['Resource'].fillna('NOT_SET')
    df.to_csv(output_dir / path.name, index=False)


test_logs = Path('results/2022-12-05_171234/logs').glob('*_test.csv')
for log in test_logs:
    correct_log(log, Path('results/2022-12-05_171234/logs_corrected'))

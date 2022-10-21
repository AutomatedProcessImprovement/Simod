import pandas as pd

df = pd.read_csv('log_metrics.csv')

result = pd.DataFrame()

for group in df.groupby(['log_name', 'metric']):
    result = pd.concat([result, pd.DataFrame({
        'log_name': [group[0][0]],
        'metric': [group[0][1]],
        'similarity': [group[1]['similarity'].mean()]
    })], ignore_index=True)

result['version'] = '3.2.0'

result.to_csv('log_metrics_mean.csv', index=False)

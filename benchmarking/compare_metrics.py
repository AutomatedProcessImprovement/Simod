from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('log_metrics_processed_concatenated.csv')

metrics = ['DL', 'LOG_MAE', 'MAE']
log_names = df['log_name'].unique()

p1 = df[df['version'] == '3.2.0'][['log_name', 'metric', 'value', 'category']]
p1['category'] = p1['category'].apply(lambda x: x + '_v3.2.0')

p2 = df[df['version'] == '3.0.0'][['log_name', 'metric', 'value', 'category']]
p2['category'] = p2['category'].apply(lambda x: x + '_v3.0.0')

pp = pd.concat([p1, p2])

Path('plots').mkdir(exist_ok=True)

for metric_group in pp.groupby('metric'):
    metric = metric_group[0]
    metric_data = metric_group[1]

    if metric not in metrics:
        continue

    group_log_names = set(log_names)
    for (log_name, log_group) in metric_data.groupby('category'):
        group_log_names = group_log_names.intersection(set(log_group['log_name']))

    data = metric_data[metric_data['log_name'].isin(group_log_names)]

    plot_data = pd.DataFrame({
        'undifferentiated_v3.0.0':
            data.where(data['category'] == 'undifferentiated_v3.0.0').dropna().sort_values('log_name')[
                'value'].tolist(),
        'undifferentiated_v3.2.0':
            data.where(data['category'] == 'undifferentiated_v3.2.0').dropna().sort_values('log_name')[
                'value'].tolist(),
        # 'differentiated_v3.2.0':
        #     data.where(data['category'] == 'differentiated_v3.2.0').dropna().sort_values('log_name')[
        #         'similarity'].tolist(),
        # 'pool_v3.2.0':
        #     data.where(data['category'] == 'pool_v3.2.0').dropna().sort_values('log_name')['similarity'].tolist(),
    }, index=sorted(group_log_names))

    plot_data.plot.bar()
    plt.title(f'{metric}')
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.7), ncol=2)
    # plt.yscale('log')
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.gcf().subplots_adjust(top=0.75)
    plt.savefig(f'plots/{metric}.png')
    # plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('log_metrics_all.csv')

for group in df.groupby('log_name'):
    sns.barplot(x='metric', y='similarity', hue='version', data=group[1])
    plt.yscale('log')
    plt.title(f'{group[0]}')
    plt.savefig(f'plots/{group[0]}.png')

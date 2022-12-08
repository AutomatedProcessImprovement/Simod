import pandas as pd


def process_metrics(metrics_file_name: str, output_file_name: str, version: str, category: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_file_name)

    result = pd.DataFrame()

    for group in df.groupby(['log_name', 'metric']):
        result = pd.concat([result, pd.DataFrame({
            'log_name': [group[0][0]],
            'metric': [group[0][1]],
            'value': [group[1]['value'].mean()]
        })], ignore_index=True)

    result['version'] = version
    result['category'] = category

    return result


# df1 = process_metrics('trial 1/log_metrics.csv', 'trial 1/pooled_log_metrics_processed.csv', version='3.2.0',
#                       category='pool')
# df2 = process_metrics('trial 1/differentiated_log_metrics.csv', 'trial 1/differentiated_log_metrics_processed.csv',
#                       version='3.2.0',
#                       category='differentiated')
df3 = process_metrics('undifferentiated_log_metrics.csv', 'undifferentiated_log_metrics_processed.csv',
                      version='3.2.0',
                      category='undifferentiated')

# df = pd.concat([df1, df2, df3], ignore_index=True)
df3.to_csv('log_metrics_processed_concatenated.csv', index=False)

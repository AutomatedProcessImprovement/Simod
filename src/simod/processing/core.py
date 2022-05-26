import pandas as pd


def remove_outliers(event_log: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(event_log, pd.DataFrame):
        raise TypeError('Event log must be a pandas DataFrame')

    # TODO: it uses specific column names, provide a more general solution

    # calculating case durations
    cases_durations = list()
    for id, trace in event_log.groupby('caseid'):
        duration = (trace['end_timestamp'].max() - trace['start_timestamp'].min()).total_seconds()
        cases_durations.append({'caseid': id, 'duration_seconds': duration})
    cases_durations = pd.DataFrame(cases_durations)

    # merging data
    event_log = event_log.merge(cases_durations, how='left', on='caseid')

    # filtering rare events
    unique_cases_durations = event_log[['caseid', 'duration_seconds']].drop_duplicates()
    first_quantile = unique_cases_durations.quantile(0.1)
    last_quantile = unique_cases_durations.quantile(0.9)
    event_log = event_log[(event_log.duration_seconds <= last_quantile.duration_seconds) & (
            event_log.duration_seconds >= first_quantile.duration_seconds)]
    event_log = event_log.drop(columns=['duration_seconds'])

    return event_log

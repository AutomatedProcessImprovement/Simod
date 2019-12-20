# -*- coding: utf-8 -*-
import pandas as pd

def define_interarrival_tasks(process_graph, log, settings):
	# Analysis of start tasks
    log = pd.DataFrame.from_records(log)
    ordering_field = 'start_timestamp'
    if settings['read_options']['one_timestamp']:
        ordering_field = 'end_timestamp'
    # Find the initial activity
    arrival_timestamps = (pd.DataFrame(log.groupby('caseid')[ordering_field].min())
                                              .reset_index()
                                              .rename(columns={ordering_field:'times'}))
    # group by day and calculate inter-arrival
    arrival_timestamps['date'] = arrival_timestamps['times'].dt.floor('d')
    inter_arrival_times = list()
    for key, group in arrival_timestamps.groupby('date'):
        daily_times = sorted(list(group.times))
        for i in range(1, len(daily_times)):
            inter_arrival_times.append((daily_times[i] - daily_times[i-1]).total_seconds())
    return inter_arrival_times
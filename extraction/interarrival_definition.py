# -*- coding: utf-8 -*-
import pandas as pd
from support_modules import support as sup

def define_interarrival_tasks(process_graph, log, settings):
	# Analysis of start tasks
    log = pd.DataFrame.from_records(log)
    ordering_field = 'start_timestamp'
    if settings['read_options']['one_timestamp']:
        ordering_field = 'end_timestamp'
    # Find the initial activity
    tasks = analize_first_tasks(process_graph)
    log = log[log.task.isin(tasks)]
    arrival_timestamps = (pd.DataFrame(log.groupby('caseid')[ordering_field].min())
                                              .reset_index()
                                              .rename(columns={ordering_field:'times'}))
    # group by day and calculate inter-arrival
    arrival_timestamps['date'] = arrival_timestamps['times'].dt.floor('d')
    inter_arrival_times = list()
    for key, group in arrival_timestamps.groupby('date'):
        daily_times = sorted(list(group.times))
        for i in range(1, len(daily_times)):
            delta = (daily_times[i] - daily_times[i-1]).total_seconds()
            # TODO: Check this condition, if interarrival is 0 what does it means?
            # if delta > 0:
            inter_arrival_times.append(delta)
    return inter_arrival_times

def analize_first_tasks(process_graph):
    tasks_list = list()
    for node in process_graph.nodes:
        if process_graph.node[node]['type']=='task':
            tasks_list.append(find_tasks_predecesors(process_graph,node))
    in_tasks = list()
    i=0
    for task in tasks_list:
        sup.print_progress(((i / (len(tasks_list)-1))* 100),'Defining inter-arrival rate ')
        for path in task['sources']:
            for in_task in path['in_tasks']:
                if process_graph.node[in_task]['type']=='start':
                    in_tasks.append(process_graph.node[task['task']]['name'])
        i+=1
    return list(set(in_tasks))

def find_tasks_predecesors(process_graph,num):
    # Sources
    r = process_graph.reverse(copy=True)
    paths = list(r.neighbors(num))
    task_paths = extract_target_tasks(r, num)
    in_paths = [sup.reduce_list(path) for path in task_paths]
    ins = [dict(in_tasks=y, in_node= x) for x,y in zip(paths, in_paths)]

    return dict(task=num,sources=ins)

def extract_target_tasks(process_graph, num):
    tasks_list=list()
    for node in process_graph.neighbors(num):
        if process_graph.node[node]['type']=='task' or process_graph.node[node]['type']=='start' or process_graph.node[node]['type']=='end':
            tasks_list.append([node])
        else:
            tasks_list.append(extract_target_tasks(process_graph, node))
    return     tasks_list
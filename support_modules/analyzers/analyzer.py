# -*- coding: utf-8 -*-
from analyzers.analyzer_ui import analyzer_interface as ui
import numpy as np
import scipy.stats as st
import support as sup
import statsmodels.stats.api as sms

from analyzers import statistics as sta

######################################
############## Support################
######################################

#def mean_confidence_interval(data, confidence=0.95):
#    a = np.array(data)
#    n = len(a)
#    m, se = np.mean(a), st.sem(a)
#    h = np.nan_to_num(se * st.t._ppf((1+confidence)/2.0, n-1))
#    return h

#def mean_confidence_interval(data, confidence=0.95):
#    a = np.array(data)
#    interval = sms.DescrStatsW(a).tconfint_mean(alpha= 1-confidence)
#    return np.abs(np.nan_to_num(np.array(interval))[1] - np.mean(a))

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    return st.sem(a)


def create_task_alias(data):
    variables = sorted(list(set([x['task'] for x in data])))
    alias = dict()
    for i in range(0, len(variables)):
        alias[variables[i]] = 'AC' + str(i + 1)
    return alias

def calculate_percentual_change(sources, tasks, alias, durations):
    data = list()
    sources.remove('log')
    for task in tasks:
        try:
            lv_processing_time = np.mean(list(filter(lambda x: x['source'] == 'log' and x['task'] == task, durations))[0]['processing_time'])
        except:
            lv_processing_time = 0
        try:
            lv_waiting_time = np.mean(list(filter(lambda x: x['source'] == 'log' and x['task'] == task, durations))[0]['waiting_time'])
        except:
            lv_waiting_time = 0
        for source in sources:
            try:
                processing_time = np.mean(list(filter(lambda x: x['source'] == source and x['task'] == task, durations))[0]['processing_time'])
            except:
                processing_time = 0
            try:
                waiting_time = np.mean(list(filter(lambda x: x['source'] == source and x['task'] == task, durations))[0]['waiting_time'])
            except:
                waiting_time = 0
            np.warnings.filterwarnings('ignore')
            if lv_processing_time != 0:
                pc_processing_time = np.round(((processing_time - lv_processing_time)/ abs(lv_processing_time)) * 100, 2)
            else:
                pc_processing_time = 0
            if lv_waiting_time != 0:
                pc_waiting_time = np.round(((waiting_time - lv_waiting_time)/ abs(lv_waiting_time)) * 100, 2)
            else:
                pc_waiting_time = 0
            if np.isnan(pc_processing_time):
                pc_processing_time = 0
            if np.isnan(pc_waiting_time):
                pc_waiting_time = 0
            data.append(dict(task=task, alias=alias[task], pc_processing_time=pc_processing_time, pc_waiting_time=pc_waiting_time, source=source))
    return data

def calculate_task_durations(sources, tasks, alias, durations):
    task_duration = list()
    for task in tasks:
        for source in sources:
            values = list(filter(lambda x: x['source'] == source and x['task'] == task, durations))
            if len(values) > 0:
                task_duration.append(dict(task=task, alias=alias[task], raw_processing=values[0]['processing_time'],processing_time = np.mean(values[0]['processing_time']),
                waiting_time = np.mean(values[0]['waiting_time']), multitasking = np.mean(values[0]['multitasking']), source=source,
                pmci=mean_confidence_interval(values[0]['processing_time']), wmci= mean_confidence_interval(values[0]['waiting_time'])))
            else:
                task_duration.append(dict(task=task, alias=alias[task], processing_time = 0,
                waiting_time = 0, multitasking = 0, source=source, pmci=0, wmci=0))
    return task_duration

def calculate_role_durations(sources, roles, role_time_use):
    role_use = list()
    for role in roles:
        for source in sources:
            values = list(filter(lambda x: x['source'] == source and x['role'] == role, role_time_use))
            if len(values) > 0:
                role_use.append(dict(role=role, raw_processing=values[0]['processing_time'],
                processing_time = np.mean(values[0]['processing_time']), source=source,
                pmci=mean_confidence_interval(values[0]['processing_time'])))
            else:
                role_use.append(dict(role=role, processing_time = 0, source=source, pmci=0))
    return role_use

def calculate_process_duration(sources, durations):
    process_durations = list()
    for source in sources:
        values = list(filter(lambda x: x['source'] == source, durations))
        if len(values) > 0:
            process_durations.append(dict( raw_processing=values[0]['processing_time'], processing_time = np.mean(values[0]['processing_time']),
            waiting_time = np.mean(values[0]['waiting_time']), multitasking = np.mean(values[0]['multitasking']), source=source,
            pmci=mean_confidence_interval(values[0]['processing_time']), wmci= mean_confidence_interval(values[0]['waiting_time'])))
        else:
            process_durations.append(dict(processing_time = 0,
            waiting_time = 0, multitasking = 0, source=source, pmci=0, wmci=0))
    return process_durations


def create_report(process_stats):
    sources =  sorted(list(set([x['source'] for x in process_stats])))
    # Tasks
    alias = create_task_alias(process_stats)
    val = sorted(alias.items())
    [print(x) for x in val]
    #IDEA las tareas deben venir directamente del modelo !
    tasks = sorted(list(set([x['task'] for x in process_stats])))
    tasks_durations = sta.task_metrics_statistics(process_stats)
    task_duration = calculate_task_durations(sources, tasks, alias, tasks_durations)
    percentual_change = calculate_percentual_change(sources, tasks, alias, tasks_durations)
    sup.create_csv_file_header(task_duration, 'task_duration.csv')

    sources =  sorted(list(set([x['source'] for x in process_stats])))
    process_durations = sta.process_metrics_statistics(process_stats)
    process_duration = calculate_process_duration(sources, process_durations)
    sup.create_csv_file_header(process_duration, 'process_duration.csv')
    # Resources

    sources =  sorted(list(set([x['source'] for x in process_stats])))
    roles=sorted(list(set([x['role'] for x in process_stats])))
    role_time_use = sta.role_statistics(process_stats)
    role_use = calculate_role_durations(sources, roles, role_time_use)
    sup.create_csv_file_header(role_use, 'role_use.csv')


    ui.analyzer_interface(task_duration, percentual_change, role_use, process_duration)

# -*- coding: utf-8 -*-
import itertools
import numpy as np
import random
import string
import jellyfish as jf
from operator import itemgetter
import itertools

def gen_mesurement(process_stats, run_num, ramp_io_perc = 0.2):
    # get log data
    log_data=list(filter(lambda x: x['source']=='log', process_stats))
    # reformat log data
    alias = create_task_alias(log_data)
    log_data = reformat_events(log_data, alias)
    # compute similarity between log and simulation runs
    similarity = list()
    for num in range(0, run_num):
        # get simulation data per run_number
        simulation_data=list(filter(lambda x: x['source']=='bimp' and x['run_num'] == num, process_stats))
        simulation_data = reformat_events(simulation_data, alias)
        # cut simulation data avoiding rampage input/output
        num_traces = int(np.round((len(log_data) * ramp_io_perc),0))
        simulation_data = simulation_data[num_traces:-num_traces]
        # select randomly the same number of log traces
        temp_log_data = random.sample(log_data, len(simulation_data))
        similarity.extend(measure_distance(temp_log_data, simulation_data, num))
    # average similarity
    process_results(similarity)

def process_results(similarity):
    data = sorted(list(similarity), key=lambda x:x['run_num'])
    run_similarity = list()
    for key, group in itertools.groupby(data, key=lambda x:x['run_num']):
        values = list(group)
        group_similarity = [x['sim_score'] for x in values]
        run_similarity.append(np.mean(group_similarity))
    print(run_similarity)
    print(np.mean(run_similarity))
    # [print(x) for x in similarity]
    
def measure_distance(log_data, simulation_data, run_num):
    similarity = list()
    for sim_instance in simulation_data:
        temp_log_data = log_data
        max_sim, max_index = 0 , 0
        for i in range(0,len(temp_log_data)):
            sim = jf.jaro_winkler(sim_instance['profile'], temp_log_data[i]['profile'])
            if max_sim < sim:
                max_sim = sim
                max_index = i
        del temp_log_data[max_index]
        similarity.append(dict(caseid=sim_instance['caseid'],sim_score=max_sim, run_num=run_num))
    return similarity

def create_task_alias(data):
    variables = sorted(list(set([x['task'] for x in data])))
    aliases = random.sample(string.ascii_letters, len(variables))
    alias = dict()
    for i in range(0, len(variables)):
        alias[variables[i]] = aliases[i]
    return alias

def reformat_events(data, alias):
    # Add alias
    [x.update(dict(alias=alias[x['task']])) for x in data]
    # Define cases
    cases = sorted(list(set([x['caseid'] for x in data])))
    # Reformat dataset
    temp_data = list()
    for case in cases:
        temp_dict= dict(caseid=case,profile='')
        events = sorted(list(filter(lambda x: x['caseid']==case, data)), key=itemgetter('start_timestamp'))
        for i in range(0, len(events)):
            temp_dict['profile'] = temp_dict['profile'] + events[i]['alias']
        temp_dict['start_time'] = events[0]['start_timestamp']
        temp_data.append(temp_dict)
    return sorted(temp_data, key=itemgetter('start_time'))

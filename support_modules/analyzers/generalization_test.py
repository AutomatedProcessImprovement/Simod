# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:37:16 2020

@author: Manuel Camargo
"""
import random
import string
import itertools
import numpy as np

from support_modules.analyzers import alpha_oracle as ao
from support_modules.analyzers.alpha_oracle import Rel

def mesurement(log, parms, ramp_io_perc = 0.2):
    random.seed(30)
    if parms['read_options']['one_timestamp']:
        log = log[['caseid', 'task', 'end_timestamp']]
        log = log.rename(columns={'end_timestamp':'timestamp'})
    else:
        log = log[['caseid', 'task', 'start_timestamp']]
        log = log.rename(columns={'start_timestamp':'timestamp'})
    tasks_alias = create_alias(log)
    assign_alias = lambda x: tasks_alias[x['task']]
    log['alias'] = log.apply(assign_alias, axis=1)
    # print(log.caseid.unique())
    footprint_matrix = ao.discover_concurrency(log, tasks_alias, True)
    # [print(k, v, sep=': ') for k, v in footprint_matrix.items()]
    
    example = create_examples(log)
    
    length = np.max([len(example['log_trace']), len(example['sim_trace'])])
    distance1 = damerau_levenshtein_distance(example) 
    print(1-(distance1/length))
    distance2 = damerau_levenshtein_distance_parallelism(example, footprint_matrix) 
    print(1-(distance2/length))

    # print(example['log_trace'])
    # for i in range(0, len(example['log_trace'])-1):
    #     print(footprint_matrix[(example['log_trace'][i],example['log_trace'][i + 1])])
    
def create_alias(log):
    """Creates char aliases for a categorical attributes.
    Args:
        quantity (int): number of aliases to create.
    Returns:
        dict: alias for a categorical attributes.
    """
    tasks = log.task.unique()
    quantity = len(tasks)
    # characters = [chr(i) for i in range(0, quantity)]
    aliases = random.sample(string.ascii_letters, quantity)
    alias = dict()
    for i in range(0, quantity):
        alias[tasks[i]] = aliases[i]
    return alias

def damerau_levenshtein_distance_parallelism(comp_sec, footprint_matrix):
    """
    Compute the Damerau-Levenshtein distance between two given
    strings (s1 and s2)
    """
    s1 = comp_sec['log_trace']
    s2 = comp_sec['sim_trace']
    p1 = comp_sec['proc_log_trace']
    p2 = comp_sec['proc_sim_trace']
    w1 = comp_sec['wait_log_trace']
    w2 = comp_sec['wait_sim_trace']
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1
    for i in range(0, lenstr1):
        for j in range(0, lenstr2):
            if s1[i] == s2[j]:
                cost = calculate_cost(p1, p2, w1, w2, i, j)
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                if footprint_matrix[(s1[i], s1[j-1])] == Rel.PARALLEL:
                    cost = calculate_cost(p1, p2, w1, w2, i, j-1)
                    print(cost)
                d[(i,j)] = min(d[(i,j)], d[i-2,j-2] + cost) # transposition
    return d[lenstr1-1,lenstr2-1]

def calculate_cost(p1, p2, w1, w2, i, j):
    t1 = p1[i] + w1[i]
    if t1 > 0:
        b1 = (p1[i]/t1)
        cost = (b1*abs(p2[j]-p1[i])) + ((1 - b1)*abs(w2[j]-w1[i]))
    else:
        cost = 0
    return cost

def damerau_levenshtein_distance(comp_sec):
    """
    Compute the Damerau-Levenshtein distance between two given
    strings (s1 and s2)
    """
    s1 = comp_sec['log_trace']
    s2 = comp_sec['sim_trace']
    p1 = comp_sec['proc_log_trace']
    p2 = comp_sec['proc_sim_trace']
    w1 = comp_sec['wait_log_trace']
    w2 = comp_sec['wait_sim_trace']
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1
    for i in range(0, lenstr1):
        for j in range(0, lenstr2):
            if s1[i] == s2[j]:
                t1 = p1[i] + w1[i]
                if t1 > 0:
                    b1 = (p1[i]/t1)
                    b2 = (w1[i]/t1)
                    cost = (b1*abs(p2[j]-p1[i])) + (b2*abs(w2[j]-w1[i]))
                else:
                    cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                            d[(i-1,j)] + 1, # deletion
                            d[(i,j-1)] + 1, # insertion
                            d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
    return d[lenstr1-1,lenstr2-1]

# =============================================================================
# Para testing unicamente
# =============================================================================
def create_examples(log):
    log = log[log.caseid.isin([107, 12])]
    log = reformat_events(log)
    comp_sec = dict()
    # comp_sec['log_trace'] = log[0]
    comp_sec['log_trace'] = ['I', 'k', 'Q', 's', 'N', 'W', 'N', 'W', 'N', 'W', 'N', 'b', 'X', 'P', 'n', 'q', 'd', 'z', 'y', 'i', 'f', 'D', 'a', 'H', 'p']
    comp_sec['sim_trace'] = ['I', 'Q', 'k', 'N', 'b', 'X', 'P', 'n', 'q', 'd', 'z', 'y', 'i', 'f', 'D', 'a', 'e', 'H']
    # comp_sec['sim_trace'] = log[1]
    comp_sec['proc_log_trace'] = [random.uniform(0.0, 1.0) for iter in range(len(log[0]))]
    comp_sec['proc_sim_trace'] = [random.uniform(0.0, 1.0) for iter in range(len(log[1]))]
    comp_sec['wait_log_trace'] = [random.uniform(0.0, 1.0) for iter in range(len(log[0]))]
    comp_sec['wait_sim_trace'] = [random.uniform(0.0, 1.0) for iter in range(len(log[1]))]
    return comp_sec
              
def reformat_events(log_df):
    """Creates series of activities, roles and relative times per trace.
    parms:
        log_df: dataframe.
    Returns:
        list: lists of activities.
    """
    temp_data = list()
    log_df = log_df.to_dict('records')
    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['timestamp']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        serie = [y['alias'] for y in trace]
        temp_data.append(serie)
    return temp_data

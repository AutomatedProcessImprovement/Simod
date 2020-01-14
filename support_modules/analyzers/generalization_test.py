# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:37:16 2020

@author: Manuel Camargo
"""
import random
import string
import itertools
import numpy as np

from collections import defaultdict


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
    alpha_concurrency = ao.discover_concurrency(log, tasks_alias, True)
    # [print(k, v, sep=': ') for k, v in alpha_concurrency.items()]
    
    example = create_examples(log)
    
    length = np.max([len(example['seqs']['s1']), len(example['seqs']['s2'])])
    
    
    # distance1 = damerau_levenshtein_distance(example) 
    # print(1-(distance1/length))
    distance2 = damerau_levenshtein_distance_parallelism(example, alpha_concurrency) 
    print(1-(distance2/length))
    # distance3 = damerau_levenshtein_distance_2(''.join(example['log_trace']),''.join(example['sim_trace']))
    # print(1-(distance3/length))
    
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

def damerau_levenshtein_distance_parallelism(comp_sec, alpha_concurrency):
    """
    Compute the Damerau-Levenshtein distance between two given
    strings (s1 and s2)
    """
    s1 = comp_sec['seqs']['s1']
    s2 = comp_sec['seqs']['s2']
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
                cost = calculate_cost(comp_sec['times'], i, j)
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                print('Compared chars:',s1[i], s2[j], sep=' ')
                if alpha_concurrency[(s1[i], s2[j])] == Rel.PARALLEL:
                    print('Chars to transpose:',s1[i], s2[j-1], sep=' ')
                    cost = calculate_cost(comp_sec['times'], i, j-1)
                print(cost)
                d[(i,j)] = min(d[(i,j)], d[i-2,j-2] + cost) # transposition
    return d[lenstr1-1,lenstr2-1]

def calculate_cost(times, s1_idx, s2_idx):
    p1 = times['p1']
    p2 = times['p2']
    w1 = times['w1']
    w2 = times['w2']
   
    t1 = p1[s1_idx] + w1[s1_idx]
    if t1 > 0:
        b1 = (p1[s1_idx]/t1)
        cost = (b1*abs(p2[s2_idx]-p1[s1_idx])) + ((1 - b1)*abs(w2[s2_idx]-w1[s1_idx]))
    else:
        cost = 0
    return cost


# =============================================================================
# Para testing unicamente
# =============================================================================
def create_examples(log):
    log = log[log.caseid.isin([107, 12])]
    log = reformat_events(log)
    comp_sec = dict()
    comp_sec['seqs'] = dict()
    comp_sec['seqs']['s1'] = ['I', 'k', 'Q', 's', 'N', 'W', 'N', 'W', 'N', 'W', 'N', 'b', 'X', 'P', 'n', 'q', 'd', 'z', 'y', 'i', 'f', 'D', 'a', 'H', 'p']
    comp_sec['seqs']['s2'] = ['I', 'Q', 'k', 'N', 's', 'X', 'P', 'n', 'q', 'd', 'z', 'y', 'i', 'f', 'D', 'a', 'e', 'H']
    comp_sec['times'] = dict()
    comp_sec['times']['p1'] = [random.uniform(0.0, 1.0) for iter in range(len(log[0]))]
    comp_sec['times']['p2'] = [random.uniform(0.0, 1.0) for iter in range(len(log[1]))]
    comp_sec['times']['w1'] = [random.uniform(0.0, 1.0) for iter in range(len(log[0]))]
    comp_sec['times']['w2'] = [random.uniform(0.0, 1.0) for iter in range(len(log[1]))]
    
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

def damerau_levenshtein_distance_2(s1, s2):

    len1 = len(s1)
    len2 = len(s2)
    infinite = len1 + len2

    # character array
    da = defaultdict(int)

    # distance matrix
    score = [[0]*(len2+2) for x in range(len1+2)]

    score[0][0] = infinite
    for i in range(0, len1+1):
        score[i+1][0] = infinite
        score[i+1][1] = i
    for i in range(0, len2+1):
        score[0][i+1] = infinite
        score[1][i+1] = i

    for i in range(1, len1+1):
        db = 0
        for j in range(1, len2+1):
            i1 = da[s2[j-1]]
            j1 = db
            cost = 1
            if s1[i-1] == s2[j-1]:
                cost = 0
                db = j

            score[i+1][j+1] = min(score[i][j] + cost,
                                  score[i+1][j] + 1,
                                  score[i][j+1] + 1,
                                  score[i1][j1] + (i-i1-1) + 1 + (j-j1-1))
        da[s1[i-1]] = i

    return score[len1+1][len2+1]
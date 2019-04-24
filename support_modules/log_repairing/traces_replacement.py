# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:54:02 2019

@author: Manuel Camargo
"""
import random
from operator import itemgetter
import jellyfish as jf
import numpy as np

def replacement(conformant, not_conformant, log):
    alias = create_task_alias(log.data)
    similarity = measure_distance(reformat_events(not_conformant, alias),
                                  reformat_events(conformant, alias))
    conformant_reformated = list()
    for trace in conformant:
        conformant_reformated.extend(trace)
    similar_traces = list()
    for similar in similarity:
        trace = list(filter(lambda x: x['caseid']==similar['sim_caseid'], conformant_reformated))
        for event in trace:
            similar_traces.append({
                    'caseid':similar['caseid']+'R',
                    'task': event['task'],
                    'start_timestamp': event['start_timestamp'],
                    'end_timestamp': event['end_timestamp'],
                    'event_type': event['event_type'], 
                    'user': event['user'], 
                    'alias': event['alias']
                    })
    conformant_reformated.extend(similar_traces)
    return conformant_reformated

def measure_distance(not_conformant, conformant):
    similarity = list()
    temp_conformant = conformant.copy()
    for not_con_trace in not_conformant:
        min_dist = jf.damerau_levenshtein_distance(not_con_trace['profile'], temp_conformant[0]['profile'])
        min_index = 0
        for i in range(0,len(temp_conformant)):
            sim = jf.damerau_levenshtein_distance(not_con_trace['profile'], temp_conformant[i]['profile'])
            if min_dist > sim:
                min_dist = sim
                min_index = i
        length=np.max([len(not_con_trace['profile']), len(temp_conformant[min_index]['profile'])])        
        similarity.append(dict(caseid=not_con_trace['caseid'],
                               sim_caseid=temp_conformant[min_index]['caseid'],
                               sim_score=(1-(min_dist/length))))
    return similarity


def create_task_alias(df):
    subsec_set = set()
    task_list = [x['task'] for x in df]
    [subsec_set.add(x) for x in task_list]
    variables = sorted(list(subsec_set))
    characters = [chr(i) for i in range(0, len(variables))]
    aliases = random.sample(characters, len(variables))
    alias = dict()
    for i, _ in enumerate(variables):
        alias[variables[i]] = aliases[i]
    return alias

def reformat_events(data, alias):
    temp_data = list()
    for case in data:
        temp_dict= dict(caseid=case[0]['caseid'], profile='')
        [x.update(dict(alias=alias[x['task']])) for x in case]
        for i in range(0, len(case)):
            temp_dict['profile'] = temp_dict['profile'] + case[i]['alias']
        temp_dict['start_time'] = case[0]['start_timestamp']
        temp_data.append(temp_dict)
    return sorted(temp_data, key=itemgetter('start_time'))

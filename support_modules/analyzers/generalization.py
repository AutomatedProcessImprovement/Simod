import numpy as np
import random
import os
from operator import itemgetter
from support_modules import support as sup

#import matplotlib.pyplot as plt

def mesurement(data, settings, rep, ramp_io_perc = 0.2):
    filtered_data = data[(data.source=='log') | ((data.source=='simulation') & (data.run_num==(rep + 1)))]
    filtered_data = filtered_data.reset_index()
    filtered_data = scaling_data(filtered_data)
    filtered_data = filtered_data.to_dict('records')
    alias = create_task_alias(filtered_data, ['task','role'])
    
    log_data = reformat_events(list(filter(lambda x: x['source']=='log', filtered_data)),
                               alias, ['task','role'])
    simulation_data = reformat_events(list(filter(lambda x: x['source']=='simulation'
                                                  and x['run_num']==(rep + 1), filtered_data)),
                               alias, ['task','role'])
    num_traces = int(len(simulation_data) * ramp_io_perc)
    simulation_data = simulation_data[num_traces:-num_traces]
    temp_log_data = random.sample(log_data, len(simulation_data))
    
    similarity = dict()
    similarity['run_num'] = (rep + 1)
#    similarity['lognorm'] = np.mean([x['sim_score'] for x in 
#                                   measure_distance(log_data, simulation_data, 'proc_lognorm')])
#    similarity['norm'] = np.mean([x['sim_score'] for x in 
#                                   measure_distance(log_data, simulation_data, 'proc_norm')])
    sim_data = measure_distance(temp_log_data, simulation_data, 'proc_act_norm')
    for x in sim_data: x['run_num'] = (rep + 1)
    similarity['act_norm'] = np.mean([x['sim_score'] for x in sim_data])
    print_measures(settings, sim_data)   
    return similarity

def print_measures(settings, measurements):
    if os.path.exists(os.path.join(os.path.join(settings['output'],
                                                'sim_data',
                                                'similarity_measures.csv'))):
        sup.create_csv_file(measurements,
                            os.path.join(os.path.join(settings['output'],
                                                'sim_data',
                                                'similarity_measures.csv')), mode='a')
    else:
        sup.create_csv_file_header(measurements,
                                   os.path.join(os.path.join(settings['output'],
                                                'sim_data',
                                                'similarity_measures.csv')))


def measure_distance(log_data, simulation_data, scale_method):
    similarity = list()
    temp_log_data = log_data.copy()
    for sim_instance in simulation_data:
        comp_sec = dict(
                log_trace=sim_instance['profile'],
                proc_log_trace=sim_instance['proc_act_norm'],
                wait_log_trace=sim_instance['wait_act_norm'],
                sim_trace=temp_log_data[0]['profile'],
                proc_sim_trace=temp_log_data[0]['proc_act_norm'],
                wait_sim_trace=temp_log_data[0]['wait_act_norm']
                )
        min_dist = damerau_levenshtein_distance(comp_sec)
        min_index = 0
        for i in range(0,len(temp_log_data)):
            comp_sec = dict(
                    log_trace=sim_instance['profile'],
                    proc_log_trace=sim_instance['proc_act_norm'],
                    wait_log_trace=sim_instance['wait_act_norm'],
                    sim_trace=temp_log_data[i]['profile'],
                    proc_sim_trace=temp_log_data[i]['proc_act_norm'],
                    wait_sim_trace=temp_log_data[i]['wait_act_norm']
                    )
            sim = damerau_levenshtein_distance(comp_sec)
            if min_dist > sim:
                min_dist = sim
                min_index = i
        length=np.max([len(sim_instance['profile']), len(temp_log_data[min_index]['profile'])])        
        similarity.append(dict(caseid=sim_instance['caseid'],
                               sim_order=sim_instance['profile'],
                               log_order=temp_log_data[min_index]['profile'],
                               sim_score=(1-(min_dist/length))))
        del temp_log_data[min_index]
    return similarity

def create_task_alias(df, features):
    subsec_set = set()
    if isinstance(features, list):
        task_list = [(x[features[0]],x[features[1]]) for x in df]   
    else:
        task_list = [x[features] for x in df]
    [subsec_set.add(x) for x in task_list]
    variables = sorted(list(subsec_set))
    characters = [chr(i) for i in range(0, len(variables))]
    aliases = random.sample(characters, len(variables))
    alias = dict()
    for i, _ in enumerate(variables):
        alias[variables[i]] = aliases[i]
    return alias

def scaling_data(data):
    summary = data.groupby(['task']).agg({'processing_time': ['mean', 'max', 'min'] }).to_dict('index')
    proc_act_norm = lambda x: x['processing_time'] / summary[x['task']][('processing_time','max')] if summary[x['task']][('processing_time','max')] >0 else 0
    data['proc_act_norm'] = data.apply(proc_act_norm, axis=1)
    #---
    summary = data.groupby(['task']).agg({'waiting_time': ['mean', 'max', 'min'] }).to_dict('index')
    wait_act_norm = lambda x: x['waiting_time'] / summary[x['task']][('waiting_time','max')] if summary[x['task']][('waiting_time','max')] >0 else 0
    data['wait_act_norm'] = data.apply(wait_act_norm, axis=1)
    return data

#def scaling_data(data):
#    #--------------
#    max_proc = np.max(data.processing_time)
#    scale = lambda x: x['processing_time']/max_proc
#    data['proc_max'] = data.apply(scale, axis=1)
#    #--------------
#    try:
#        logit = lambda x: math.log1p(x['processing_time'])
#        data['proc_log'] = data.apply(logit, axis=1)
#        max_proc = np.max(data.proc_log)
#        norm = lambda x: x['proc_log']/max_proc
#        data['proc_lognorm'] = data.apply(norm, axis=1)
#    except Exception as e:
#        print(str(e))
#    #--------------
#    max_proc = np.max(data.processing_time)
#    min_proc = np.min(data.processing_time)
#    norm = lambda x: (x['processing_time'] - min_proc) / (max_proc - min_proc)
#    data['proc_norm'] = data.apply(norm, axis=1)
#    #--------------
#    summary = data.groupby(['task']).agg({'processing_time': ['mean', 'max', 'min'] }).to_dict('index')
#    act_norm = lambda x: (x['processing_time'] - summary[x['task']][('processing_time','min')])/(summary[x['task']][('processing_time','max')] - summary[x['task']][('processing_time','min')])
#    act_norm = lambda x: x['processing_time'] / summary[x['task']][('processing_time','max')] if summary[x['task']][('processing_time','max')] >0 else 0
#    data['proc_act_norm'] = data.apply(act_norm, axis=1)
#    #--------------
#    fig, axes = plt.subplots(nrows=1, ncols=4)
#    data.proc_max.plot.hist(bins=12, alpha=0.5, ax=axes[0])
#    data.proc_lognorm.plot.hist(bins=12, alpha=0.5, ax=axes[1])
#    data.proc_norm.plot.hist(bins=12, alpha=0.5, ax=axes[2])
#    data.proc_act_norm.plot.hist(bins=12, alpha=0.5, ax=axes[3])
#    plt.show()
#    return data

def reformat_events(data, alias, features):
    # Add alias
    if isinstance(features, list):
        [x.update(dict(alias=alias[(x[features[0]],x[features[1]])])) for x in data]   
    else:
        [x.update(dict(alias=alias[x[features]])) for x in data]
    # Define cases
    cases = sorted(list(set([x['caseid'] for x in data])))
    # Reformat dataset
    temp_data = list()
    for case in cases:
        temp_dict= dict(caseid=case,profile='', processing=list(), proc_act_norm=list(),
                        waiting=list(), wait_act_norm=list())
        events = sorted(list(filter(lambda x: x['caseid']==case, data)), key=itemgetter('start_timestamp'))
        for i in range(0, len(events)):
            temp_dict['profile'] = temp_dict['profile'] + events[i]['alias']
            temp_dict['processing'].append(events[i]['processing_time'])
            temp_dict['proc_act_norm'].append(events[i]['proc_act_norm'])
            temp_dict['waiting'].append(events[i]['waiting_time'])
            temp_dict['wait_act_norm'].append(events[i]['wait_act_norm'])
        temp_dict['start_time'] = events[0]['start_timestamp']
        temp_data.append(temp_dict)
    return sorted(temp_data, key=itemgetter('start_time'))


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
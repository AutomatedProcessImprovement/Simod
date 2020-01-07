# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:03:15 2020

@author: Manuel Camargo
"""
import itertools
from enum import Enum

class Rel(Enum):
    FOLLOWS = 1
    PRECEDES = 2
    NOT_CONNECTED = 3
    PARALLEL = 4

def discover_concurrency(log, tasks_alias):
    seq_flows = reformat_events(log)
    freqs = count_freq(seq_flows)
    # Create footprint matrix
    footprint_matrix = dict()
    for source in tasks_alias.values():
        for target in tasks_alias.values():
            footprint_matrix[(source, target)] = Rel.NOT_CONNECTED
    # Fill footprint matrix
    for relation in freqs:
        if footprint_matrix[relation] == Rel.NOT_CONNECTED:
            footprint_matrix[relation] = Rel.PRECEDES
            footprint_matrix[(relation[1],relation[0])] = Rel.FOLLOWS
        elif footprint_matrix[relation] == Rel.FOLLOWS:
            footprint_matrix[relation] = Rel.PARALLEL
            footprint_matrix[(relation[1],relation[0])] = Rel.PARALLEL
    return footprint_matrix

def count_freq(seq_flows):
    freqs = dict()
    for flow in seq_flows:
        for i in range(0, len(flow)-1):
            if (flow[i],flow[i+1]) in freqs:
                freqs[(flow[i],flow[i+1])] += 1
            else:
                freqs[(flow[i],flow[i+1])] = 1
    return freqs

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


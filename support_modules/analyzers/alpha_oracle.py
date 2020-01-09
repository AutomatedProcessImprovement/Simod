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

class AlphaOracle(object):
    """
	This class provides the alpha concurrency oracle information
	"""
    def __init__(self, log, tasks_alias, one_timestamp, look_for_loops=False):
        """constructor"""
        self.log = log
        self.tasks_alias = tasks_alias
        self.one_timestamp = one_timestamp
        self.look_for_loops = look_for_loops
        self.oracle = self.discover_concurrency()
        
    def discover_concurrency(self):
        seq_flows = self.reformat_events()
        freqs = self.count_freq(seq_flows)
        # Create footprint matrix
        footprint_matrix = dict()
        for source in self.tasks_alias.values():
            for target in self.tasks_alias.values():
                footprint_matrix[(source, target)] = Rel.NOT_CONNECTED
        # Fill footprint matrix
        for relation in freqs:
            if footprint_matrix[relation] == Rel.NOT_CONNECTED:
                footprint_matrix[relation] = Rel.PRECEDES
                footprint_matrix[(relation[1],relation[0])] = Rel.FOLLOWS
            elif footprint_matrix[relation] == Rel.FOLLOWS:
                footprint_matrix[relation] = Rel.PARALLEL
                footprint_matrix[(relation[1],relation[0])] = Rel.PARALLEL
                
        if self.look_for_loops:
            for seq in seq_flows:
                for i in range(0, len(seq)-2, 2):
                    if seq[i] == seq[i + 2]:
                        footprint_matrix[(seq[i], seq[i+1])] = Rel.PRECEDES
                        footprint_matrix[(seq[i+1], seq[i])] = Rel.PRECEDES
        return footprint_matrix
    
    def count_freq(self, seq_flows):
        freqs = dict()
        for flow in seq_flows:
            for i in range(0, len(flow)-1):
                if (flow[i],flow[i+1]) in freqs:
                    freqs[(flow[i],flow[i+1])] += 1
                else:
                    freqs[(flow[i],flow[i+1])] = 1
        return freqs
    
    def reformat_events(self):
        """Creates series of activities, roles and relative times per trace.
        parms:
            log_df: dataframe.
        Returns:
            list: lists of activities.
        """
        temp_data = list()
        temp_df = self.log.copy()
        alias = lambda x: self.tasks_alias[x['task']]
        temp_df['alias'] = temp_df.apply(alias, axis=1)
        self.log = temp_df
        log_df = self.log.to_dict('records')
        if self.one_timestamp:
            log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
        else:
            log_df = sorted(log_df, key=lambda x: (x['caseid'], x['start_timestamp']))
            
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            serie = [y['alias'] for y in trace]
            temp_data.append(serie)
        return temp_data
    

import random
import os
import itertools

from operator import itemgetter
import numpy as np

from scipy.optimize import linear_sum_assignment
from support_modules import support as sup

import support_modules.analyzers.alpha_oracle as ao
from support_modules.analyzers.alpha_oracle import Rel

class SimilarityEvaluator(object):
    """
        This class evaluates the similarity of two event-logs
     """
    def __init__(self, data, settings, rep, ramp_io_perc = 0.2):
        """constructor"""
        self.output = settings['output']
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.rep_num = rep + 1
        self.ramp_io_perc = ramp_io_perc
        
        self.data = self.scaling_data(data[(data.source=='log') | ((data.source=='simulation') & (data.run_num==self.rep_num))])
        
        self.log_data = self.data[self.data.source=='log']
        self.simulation_data = self.data[(self.data.source=='simulation') & (self.data.run_num==self.rep_num)]
        self.alias = self.create_task_alias('task')
        
        self.alpha_concurrency = ao.AlphaOracle(self.log_data, self.alias, self.one_timestamp, True)
        
        self.measures = self.mesurement()
        self.similarity = {'run_num': rep + 1}
        self.similarity['act_norm'] = np.mean([x['sim_score'] for x in self.measures])
        

    def mesurement(self):
        # Inputs reformating 
        log_data = self.reformat_events(self.log_data.to_dict('records'), 'task')
        simulation_data = self.reformat_events(self.simulation_data.to_dict('records'), 'task')
        # Ramp i/o percentage
        num_traces = int(len(simulation_data) * self.ramp_io_perc)
        simulation_data = simulation_data[num_traces:-num_traces]
        sampled_log_data = random.sample(log_data, len(simulation_data))
        # similarity measurement and matching
        return self.measure_distance(sampled_log_data, simulation_data)

    
    def print_measures(self):
        if os.path.exists(os.path.join(os.path.join(self.output,
                                                    'sim_data',
                                                    'similarity_measures.csv'))):
            sup.create_csv_file(self.measures,
                                os.path.join(os.path.join(self.output,
                                                    'sim_data',
                                                    'similarity_measures.csv')), mode='a')
        else:
            sup.create_csv_file_header(self.measures,
                                       os.path.join(os.path.join(self.output,
                                                    'sim_data',
                                                    'similarity_measures.csv')))
   
    def measure_distance(self, log_data, simulation_data):
        similarity = list()
        matrix_len = len(log_data)
        cost_matrix = [[0 for c in range(matrix_len)] for r in range(matrix_len)]
        # Create cost matrix
        # start = timer()
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                comp_sec = self.create_comparison_elements(simulation_data, log_data, i, j)
                length=np.max([len(comp_sec['seqs']['s1']), len(comp_sec['seqs']['s2'])])
                distance = self.timed_string_distance_alpha(comp_sec, self.alpha_concurrency.oracle)/length
                cost_matrix[i][j] = distance
        # end = timer()
        # print(end - start)
        # Matching using the hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
        # Create response
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx]['caseid'],
                                sim_order=simulation_data[idx]['profile'],
                                log_order=log_data[idy]['profile'],
                                sim_score=(1-(cost_matrix[idx][idy]))))
        return similarity

    def create_comparison_elements(self, serie1, serie2, id1, id2):
        comp_sec = dict()
        comp_sec['seqs'] = dict()
        comp_sec['seqs']['s1'] = serie1[id1]['profile']
        comp_sec['seqs']['s2'] = serie2[id2]['profile']
        comp_sec['times'] = dict()
        if self.one_timestamp:
            comp_sec['times']['p1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p2'] = serie2[id2]['dur_act_norm']
        else:
            comp_sec['times']['p1'] = serie1[id1]['proc_act_norm']
            comp_sec['times']['p2'] = serie2[id2]['proc_act_norm']
            comp_sec['times']['w1'] = serie1[id1]['wait_act_norm']
            comp_sec['times']['w2'] = serie2[id2]['wait_act_norm']
        return comp_sec
    
    def create_task_alias(self, features):
        df = self.data.to_dict('records')
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
    
    def scaling_data(self, data):
        if self.one_timestamp:
            summary = data.groupby(['task']).agg({'duration': ['mean', 'max', 'min'] }).to_dict('index')
            dur_act_norm = lambda x: x['duration'] / summary[x['task']][('duration','max')] if summary[x['task']][('duration','max')] >0 else 0
            data['dur_act_norm'] = data.apply(dur_act_norm, axis=1)
        else:
            summary = data.groupby(['task']).agg({'processing_time': ['mean', 'max', 'min'] }).to_dict('index')
            proc_act_norm = lambda x: x['processing_time'] / summary[x['task']][('processing_time','max')] if summary[x['task']][('processing_time','max')] >0 else 0
            data['proc_act_norm'] = data.apply(proc_act_norm, axis=1)
            #---
            summary = data.groupby(['task']).agg({'waiting_time': ['mean', 'max', 'min'] }).to_dict('index')
            wait_act_norm = lambda x: x['waiting_time'] / summary[x['task']][('waiting_time','max')] if summary[x['task']][('waiting_time','max')] >0 else 0
            data['wait_act_norm'] = data.apply(wait_act_norm, axis=1)
        return data
    
    def reformat_events(self, data, features):
        """Creates series of activities, roles and relative times per trace.
        parms:
            log_df: dataframe.
            ac_table (dict): index of activities.
            rl_table (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=self.alias[(x[features[0]],x[features[1]])])) for x in data]   
        else:
            [x.update(dict(alias=self.alias[x[features]])) for x in data]
        temp_data = list()
        # define ordering keys and columns
        if self.one_timestamp:
            columns = ['alias', 'duration', 'dur_act_norm']
            sort_key = 'end_timestamp'
        else:
            sort_key = 'start_timestamp'
            columns = ['alias', 'processing_time', 'proc_act_norm', 'waiting_time', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'alias':
                    temp_dict = {**{'profile': serie},**temp_dict}
                else:
                    serie = [y[x] for y in trace]
                temp_dict = {**{x: serie},**temp_dict}
            temp_dict = {**{'caseid':key, 'start_time': trace[0][sort_key]},**temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))
    
    def timed_string_distance_alpha(self, comp_sec, alpha_concurrency):
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
                    cost = self.calculate_cost(comp_sec['times'], i, j)
                else:
                    cost = 1
                d[(i,j)] = min(
                               d[(i-1,j)] + 1, # deletion
                               d[(i,j-1)] + 1, # insertion
                               d[(i-1,j-1)] + cost, # substitution
                              )
                if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                    if alpha_concurrency[(s1[i], s2[j])] == Rel.PARALLEL:
                        cost = self.calculate_cost(comp_sec['times'], i, j-1)
                    d[(i,j)] = min(d[(i,j)], d[i-2,j-2] + cost) # transposition
        return d[lenstr1-1,lenstr2-1]
    
    def calculate_cost(self, times, s1_idx, s2_idx):
        if self.one_timestamp:
            p1 = times['p1']
            p2 = times['p2']
            cost = np.abs(p2[s2_idx]-p1[s1_idx]) if p1[s1_idx] > 0 else 0
        else:
            p1 = times['p1']
            p2 = times['p2']
            w1 = times['w1']
            w2 = times['w2']
            t1 = p1[s1_idx] + w1[s1_idx]
            if t1 > 0:
                b1 = (p1[s1_idx]/t1)
                cost = (b1*np.abs(p2[s2_idx]-p1[s1_idx])) + ((1 - b1)*np.abs(w2[s2_idx]-w1[s1_idx]))
            else:
                cost = 0
        return cost

    # def damerau_levenshtein_distance(comp_sec):
    #     """
    #     Compute the Damerau-Levenshtein distance between two given
    #     strings (s1 and s2)
    #     """
    #     s1 = comp_sec['log_trace']
    #     s2 = comp_sec['sim_trace']
    #     p1 = comp_sec['proc_log_trace']
    #     p2 = comp_sec['proc_sim_trace']
    #     w1 = comp_sec['wait_log_trace']
    #     w2 = comp_sec['wait_sim_trace']
    #     d = {}
    #     lenstr1 = len(s1)
    #     lenstr2 = len(s2)
    #     for i in range(-1,lenstr1+1):
    #         d[(i,-1)] = i+1
    #     for j in range(-1,lenstr2+1):
    #         d[(-1,j)] = j+1
    #     for i in range(0, lenstr1):
    #         for j in range(0, lenstr2):
    #             if s1[i] == s2[j]:
    #                 t1 = p1[i] + w1[i]
    #                 if t1 > 0:
    #                     b1 = (p1[i]/t1)
    #                     b2 = (w1[i]/t1)
    #                     cost = (b1*abs(p2[j]-p1[i])) + (b2*abs(w2[j]-w1[i]))
    #                 else:
    #                     cost = 0
    #             else:
    #                 cost = 1
    #             d[(i,j)] = min(
    #                            d[(i-1,j)] + 1, # deletion
    #                            d[(i,j-1)] + 1, # insertion
    #                            d[(i-1,j-1)] + cost, # substitution
    #                           )
    #             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
    #                 d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
    #     return d[lenstr1-1,lenstr2-1]

# =============================================================================
# Test kernel
# =============================================================================
# evaluation = SimilarityEvaluator(data, settings, rep)
# print(evaluation.similarity)

"""
Created on Fri Jan 10 11:40:46 2020

@author: Manuel Camargo
"""
import random
import os
import itertools
from operator import itemgetter

import jellyfish as jf
import numpy as np
from scipy.optimize import linear_sum_assignment

from support_modules import support as sup
from support_modules.analyzers import alpha_oracle as ao
from support_modules.analyzers.alpha_oracle import Rel


class SimilarityEvaluator():
    """
        This class evaluates the similarity of two event-logs
     """

    def __init__(self, data, settings, rep, metric='tsd'):
        """constructor"""
        self.output = settings['output']
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.rep_num = rep + 1
        # posible metrics tsd, dl_mae
        self.metric = metric
        self.ramp_io_perc = 0.2

        self.data = self.scaling_data(
            data[(data.source == 'log') |
                 ((data.source == 'simulation') &
                  (data.run_num == self.rep_num))])

        self.log_data = self.data[self.data.source == 'log']
        self.simulation_data = self.data[(self.data.source == 'simulation') &
                                         (self.data.run_num == self.rep_num)]
        self.alias = self.create_task_alias('task')

        self.alpha_concurrency = ao.AlphaOracle(self.log_data,
                                                self.alias,
                                                self.one_timestamp, True)

        self.measures = self.mesurement()
        self.similarity = {'run_num': rep + 1}
        self.similarity['act_norm'] = np.mean([x['sim_score']
                                               for x in self.measures])

    def mesurement(self):
        """
        Measures the distance of two event-logs with with tsd or dl and mae distance

        Returns
        -------
        distance : float

        """
        # Inputs reformating
        log_data = self.reformat_events(
            self.log_data.to_dict('records'), 'task')
        simulation_data = self.reformat_events(
            self.simulation_data.to_dict('records'), 'task')
        # Ramp i/o percentage
        num_traces = int(len(simulation_data) * self.ramp_io_perc)
        simulation_data = simulation_data[num_traces:-num_traces]
        sampled_log_data = random.sample(log_data, len(simulation_data))
        # similarity measurement and matching
        if self.metric == 'tsd':
            distance = self.tsd_distance(sampled_log_data, simulation_data)
        else:
            distance = self.dl_mae_distance(sampled_log_data, simulation_data)
        return distance

    def print_measures(self):
        """
        Prints the similarity results detail
        """
        print_path = os.path.join(self.output, 'sim_data', 'measures.csv')
        if os.path.exists(print_path):
            sup.create_csv_file(self.measures, print_path, mode='a')
        else:
            sup.create_csv_file_header(self.measures, print_path)

# =============================================================================
# Timed string distance
# =============================================================================

    def tsd_distance(self, log_data, simulation_data):
        """
        Timed string distance calculation

        Parameters
        ----------
        log_data : Ground truth list
        simulation_data : List

        Returns
        -------
        similarity : tsd similarity

        """
        similarity = list()
        matrix_len = len(log_data)
        cost_matrix = [[0 for c in range(matrix_len)] for r in range(matrix_len)]
        # Create cost matrix
        # start = timer()
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                comp_sec = self.create_comparison_elements(simulation_data,
                                                           log_data, i, j)
                length = np.max([len(comp_sec['seqs']['s_1']),
                                 len(comp_sec['seqs']['s_2'])])
                distance = self.timed_string_distance_alpha(comp_sec,
                                                            self.alpha_concurrency.oracle)/length
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
        """
        Creates a dictionary of the elements to compare

        Parameters
        ----------
        serie1 : List
        serie2 : List
        id1 : integer
        id2 : integer

        Returns
        -------
        comp_sec : dictionary of comparison elements

        """
        comp_sec = dict()
        comp_sec['seqs'] = dict()
        comp_sec['seqs']['s_1'] = serie1[id1]['profile']
        comp_sec['seqs']['s_2'] = serie2[id2]['profile']
        comp_sec['times'] = dict()
        if self.one_timestamp:
            comp_sec['times']['p_1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['dur_act_norm']
        else:
            comp_sec['times']['p_1'] = serie1[id1]['proc_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['proc_act_norm']
            comp_sec['times']['w_1'] = serie1[id1]['wait_act_norm']
            comp_sec['times']['w_2'] = serie2[id2]['wait_act_norm']
        return comp_sec

    def timed_string_distance_alpha(self, comp_sec, alpha_concurrency):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s_1 and s_2)

        Parameters
        ----------
        comp_sec : TYPE
            DESCRIPTION.
        alpha_concurrency : TYPE
            DESCRIPTION.

        Returns
        -------
        Float
        """
        s_1 = comp_sec['seqs']['s_1']
        s_2 = comp_sec['seqs']['s_2']
        dist = {}
        lenstr1 = len(s_1)
        lenstr2 = len(s_2)
        for i in range(-1, lenstr1+1):
            dist[(i, -1)] = i+1
        for j in range(-1, lenstr2+1):
            dist[(-1, j)] = j+1
        for i in range(0, lenstr1):
            for j in range(0, lenstr2):
                if s_1[i] == s_2[j]:
                    cost = self.calculate_cost(comp_sec['times'], i, j)
                else:
                    cost = 1
                dist[(i, j)] = min(
                    dist[(i-1, j)] + 1, # deletion
                    dist[(i, j-1)] + 1, # insertion
                    dist[(i-1, j-1)] + cost # substitution
                    )
                if i and j and s_1[i] == s_2[j-1] and s_1[i-1] == s_2[j]:
                    if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                        cost = self.calculate_cost(comp_sec['times'], i, j-1)
                    dist[(i, j)] = min(dist[(i, j)], dist[i-2, j-2] + cost)  # transposition
        return dist[lenstr1-1, lenstr2-1]

    def calculate_cost(self, times, s1_idx, s2_idx):
        """


        Parameters
        ----------
        times : TYPE
            DESCRIPTION.
        s1_idx : TYPE
            DESCRIPTION.
        s2_idx : TYPE
            DESCRIPTION.

        Returns
        -------
        cost : TYPE
            DESCRIPTION.

        """
        if self.one_timestamp:
            p_1 = times['p_1']
            p_2 = times['p_2']
            cost = np.abs(p_2[s2_idx]-p_1[s1_idx]) if p_1[s1_idx] > 0 else 0
        else:
            p_1 = times['p_1']
            p_2 = times['p_2']
            w_1 = times['w_1']
            w_2 = times['w_2']
            t_1 = p_1[s1_idx] + w_1[s1_idx]
            if t_1 > 0:
                b_1 = (p_1[s1_idx]/t_1)
                cost = ((b_1*np.abs(p_2[s2_idx]-p_1[s1_idx])) +
                        ((1 - b_1)*np.abs(w_2[s2_idx]-w_1[s1_idx])))
            else:
                cost = 0
        return cost

# =============================================================================
# dl and mae distance
# =============================================================================
    def dl_mae_distance(self, log_data, simulation_data):
        """


        Parameters
        ----------
        log_data : TYPE
            DESCRIPTION.
        simulation_data : TYPE
            DESCRIPTION.

        Returns
        -------
        similarity : TYPE
            DESCRIPTION.

        """
        similarity = list()
        matrix_len = len(log_data)
        dl_matrix = [[0 for c in range(matrix_len)] for r in range(matrix_len)]
        mae_matrix = [[0 for c in range(matrix_len)] for r in range(matrix_len)]
        # Create cost matrix
        # start = timer()
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                d_l, mae = self.calculate_distances(simulation_data, log_data, i, j)
                dl_matrix[i][j] = d_l
                mae_matrix[i][j] = mae
        # end = timer()
        # print(end - start)
        dl_matrix = np.array(dl_matrix)
        mae_matrix = np.array(mae_matrix)
        # MAE normalized
        max_mae = mae_matrix.max()
        mae_matrix = np.divide(mae_matrix, max_mae)
        # multiple both matrixes by Beta equal to 0.5
        dl_matrix = np.multiply(dl_matrix, 0.5)
        mae_matrix = np.multiply(mae_matrix, 0.5)
        # add each point in between
        cost_matrix = np.add(dl_matrix, mae_matrix)
        # Matching using the hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
        # Create response
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx]['caseid'],
                                   sim_order=simulation_data[idx]['profile'],
                                   log_order=log_data[idy]['profile'],
                                   sim_score=(1-(cost_matrix[idx][idy]))))
        return similarity

    def calculate_distances(self, serie1, serie2, id1, id2):
        """


        Parameters
        ----------
        serie1 : TYPE
            DESCRIPTION.
        serie2 : TYPE
            DESCRIPTION.
        id1 : TYPE
            DESCRIPTION.
        id2 : TYPE
            DESCRIPTION.

        Returns
        -------
        dl : TYPE
            DESCRIPTION.
        mae : TYPE
            DESCRIPTION.

        """
        length = np.max([len(serie1[id1]['profile']),
                         len(serie2[id2]['profile'])])
        d_l = jf.damerau_levenshtein_distance(''.join(serie1[id1]['profile']),
                                              ''.join(serie2[id2]['profile']))/length

        cicle_time_s1 = (serie1[id1]['end_time'] - serie1[id1]['start_time']).total_seconds()
        cicle_time_s2 = (serie2[id2]['end_time'] - serie2[id2]['start_time']).total_seconds()
        mae = np.mean(np.abs(cicle_time_s1 - cicle_time_s2))
        return d_l, mae


# =============================================================================
# Support methods
# =============================================================================

    def create_task_alias(self, features):
        """


        Parameters
        ----------
        features : TYPE
            DESCRIPTION.

        Returns
        -------
        alias : TYPE
            DESCRIPTION.

        """
        data = self.data.to_dict('records')
        subsec_set = set()
        if isinstance(features, list):
            task_list = [(x[features[0]], x[features[1]]) for x in data]
        else:
            task_list = [x[features] for x in data]
        [subsec_set.add(x) for x in task_list]
        variables = sorted(list(subsec_set))
        characters = [chr(i) for i in range(0, len(variables))]
        aliases = random.sample(characters, len(variables))
        alias = dict()
        for i, _ in enumerate(variables):
            alias[variables[i]] = aliases[i]
        return alias

    def scaling_data(self, data):
        """


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        if self.one_timestamp:
            summary = data.groupby(['task']).agg(
                {'duration': ['mean', 'max', 'min']}).to_dict('index')
            dur_act_norm = (lambda x: x['duration'] / summary[x['task']][('duration', 'max')] if summary[x['task']][('duration', 'max')] > 0 else 0)
            data['dur_act_norm'] = data.apply(dur_act_norm, axis=1)
        else:
            summary = data.groupby(['task']).agg({'processing_time': ['mean', 'max', 'min']}).to_dict('index')
            proc_act_norm = lambda x: x['processing_time'] / summary[x['task']][('processing_time', 'max')] if summary[x['task']][('processing_time', 'max')] > 0 else 0
            data['proc_act_norm'] = data.apply(proc_act_norm, axis=1)
            #---
            summary = data.groupby(['task']).agg({'waiting_time': ['mean', 'max', 'min']}).to_dict('index')
            wait_act_norm = lambda x: x['waiting_time'] / summary[x['task']][('waiting_time', 'max')] if summary[x['task']][('waiting_time', 'max')] > 0 else 0
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
            [x.update(dict(alias=self.alias[(x[features[0]], x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=self.alias[x[features]])) for x in data]
        temp_data = list()
        # define ordering keys and columns
        if self.one_timestamp:
            columns = ['alias', 'duration', 'dur_act_norm']
            sort_key = 'end_timestamp'
        else:
            sort_key = 'start_timestamp'
            columns = ['alias', 'processing_time',
                       'proc_act_norm', 'waiting_time', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {**{col: serie}, **temp_dict}
            temp_dict = {**{'caseid': key, 'start_time': trace[0][sort_key],
                            'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))

    # def damerau_levenshtein_distance(comp_sec):
    #     """
    #     Compute the Damerau-Levenshtein distance between two given
    #     strings (s_1 and s_2)
    #     """
    #     s_1 = comp_sec['log_trace']
    #     s_2 = comp_sec['sim_trace']
    #     p_1 = comp_sec['proc_log_trace']
    #     p_2 = comp_sec['proc_sim_trace']
    #     w_1 = comp_sec['wait_log_trace']
    #     w_2 = comp_sec['wait_sim_trace']
    #     d = {}
    #     lenstr1 = len(s_1)
    #     lenstr2 = len(s_2)
    #     for i in range(-1,lenstr1+1):
    #         d[(i,-1)] = i+1
    #     for j in range(-1,lenstr2+1):
    #         d[(-1,j)] = j+1
    #     for i in range(0, lenstr1):
    #         for j in range(0, lenstr2):
    #             if s_1[i] == s_2[j]:
    #                 t_1 = p_1[i] + w_1[i]
    #                 if t_1 > 0:
    #                     b_1 = (p_1[i]/t_1)
    #                     b2 = (w_1[i]/t_1)
    #                     cost = (b_1*abs(p_2[j]-p_1[i])) + (b2*abs(w_2[j]-w_1[i]))
    #                 else:
    #                     cost = 0
    #             else:
    #                 cost = 1
    #             d[(i,j)] = min(
    #                            d[(i-1,j)] + 1, # deletion
    #                            d[(i,j-1)] + 1, # insertion
    #                            d[(i-1,j-1)] + cost, # substitution
    #                           )
    #             if i and j and s_1[i]==s_2[j-1] and s_1[i-1] == s_2[j]:
    #                 d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
    #     return d[lenstr1-1,lenstr2-1]

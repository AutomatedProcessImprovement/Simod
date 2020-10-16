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

from scipy.stats import wasserstein_distance


from support_modules import support as sup
from support_modules.analyzers import alpha_oracle as ao
from support_modules.analyzers.alpha_oracle import Rel

# import support as sup
# import alpha_oracle as ao
# from alpha_oracle import Rel

import pandas as pd
# import json

##%%

class SimilarityEvaluator():
    """
        This class evaluates the similarity of two event-logs
     """

    def __init__(self, data, settings, rep):
        """constructor"""
        self.output = settings['output']
        self.one_timestamp = settings['read_options']['one_timestamp']
        self.rep_num = rep + 1
        # posible metrics tsd, dl_mae, tsd_min
        self.ramp_io_perc = 0.2

        self.data = self.scaling_data(
            data[(data.source == 'log') |
                 ((data.source == 'simulation') &
                  (data.run_num == self.rep_num))])
        # load data
        self.log_data = self.data[self.data.source == 'log']
        self.simulation_data = self.data[(self.data.source == 'simulation') &
                                         (self.data.run_num == self.rep_num)]
        self.alias = self.create_task_alias('task')

        self.alpha_concurrency = ao.AlphaOracle(self.log_data,
                                                self.alias,
                                                self.one_timestamp, True)
        # reformat and sampling data
        self.log_data = self.reformat_events(
            self.log_data.to_dict('records'), 'task')
        self.simulation_data = self.reformat_events(
            self.simulation_data.to_dict('records'), 'task')
        num_traces = int(len(self.simulation_data) * self.ramp_io_perc)
        self.simulation_data = self.simulation_data[num_traces:-num_traces]
        self.log_data = random.sample(self.log_data, len(self.simulation_data))

    def measure_distance(self, metric):
        """
        Measures the distance of two event-logs
        with with tsd or dl and mae distance

        Returns
        -------
        distance : float

        """
        # similarity measurement and matching
        evaluator = self._get_evaluator(metric)
        if metric in ['day_emd', 'day_hour_emd', 'cal_emd']:
            distance = evaluator(self.log_data, 
                                 self.simulation_data, 
                                 criteria=metric)
        else:
            distance = evaluator(self.log_data, self.simulation_data)
        self.similarity = {'run_num': self.rep_num,
                           'metric': metric,
                           'sim_val': np.mean(
                               [x['sim_score'] for x in distance])}
        

    def _get_evaluator(self, metric):
        if metric == 'tsd':
            return self.tsd_metric
        elif metric == 'tsd_min':
            return self.tsd_min_pattern
        elif metric == 'mae':
            return self.mae_metric
        elif metric in ['hour_emd', 'day_emd', 'day_hour_emd', 'cal_emd']:
            return self.log_emd_metric
        elif metric == 'dl_mae':
            return self.dl_mae_distance
        else:
            raise ValueError(metric)

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

    def tsd_metric(self, log_data, simulation_data):
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
        mx_len = len(log_data)
        cost_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
        # Create cost matrix
        for i in range(0, mx_len):
            for j in range(0, mx_len):
                comp_sec = self.create_comparison_elements(simulation_data,
                                                           log_data, i, j)
                length = np.max([len(comp_sec['seqs']['s_1']),
                                 len(comp_sec['seqs']['s_2'])])
                distance = self.tsd_alpha(comp_sec,
                                          self.alpha_concurrency.oracle)/length
                cost_matrix[i][j] = distance
        # Matching using the hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
        # Create response
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx]['caseid'],
                                   sim_order=simulation_data[idx]['profile'],
                                   log_order=log_data[idy]['profile'],
                                   sim_score=(1-(cost_matrix[idx][idy]))))
        return similarity

    def tsd_min_pattern(self, log_data, simulation_data):
        similarity = list()
        temp_log_data = log_data.copy()
        for i in range(0, len(simulation_data)):
            comp_sec = self.create_comparison_elements(simulation_data,
                                                       temp_log_data, i, 0)
            min_dist = self.tsd_alpha(comp_sec, self.alpha_concurrency.oracle)
            min_idx = 0
            for j in range(1, len(temp_log_data)):
                comp_sec = self.create_comparison_elements(simulation_data,
                                                           temp_log_data, i, j)
                sim = self.tsd_alpha(comp_sec, self.alpha_concurrency.oracle)
                if min_dist > sim:
                    min_dist = sim
                    min_idx = j
            length = np.max([len(simulation_data[i]['profile']),
                             len(temp_log_data[min_idx]['profile'])])
            similarity.append(dict(caseid=simulation_data[i]['caseid'],
                                   sim_order=simulation_data[i]['profile'],
                                   log_order=temp_log_data[min_idx]['profile'],
                                   sim_score=(1-(min_dist/length))))
            del temp_log_data[min_idx]
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

    def tsd_alpha(self, comp_sec, alpha_concurrency):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s_1 and s_2)
        Parameters
        ----------
        comp_sec : dict
        alpha_concurrency : dict
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
        Takes two events and calculates the penalization based on mae distance

        Parameters
        ----------
        times : dict with lists of times
        s1_idx : integer
        s2_idx : integer

        Returns
        -------
        cost : float
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
        similarity score

        Parameters
        ----------
        log_data : list of events
        simulation_data : list simulation event log

        Returns
        -------
        similarity : float

        """
        similarity = list()
        mx_len = len(log_data)
        dl_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
        mae_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
        # Create cost matrix
        # start = timer()
        for i in range(0, mx_len):
            for j in range(0, mx_len):
                d_l, ae = self._calculate_distances(
                    simulation_data, log_data, i, j)
                dl_matrix[i][j] = d_l
                mae_matrix[i][j] = ae
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

    def _calculate_distances(self, serie1, serie2, id1, id2):
        """


        Parameters
        ----------
        serie1 : list
        serie2 : list
        id1 : index of the list 1
        id2 : index of the list 2

        Returns
        -------
        dl : float value
        ae : absolute error value
        """
        length = np.max([len(serie1[id1]['profile']),
                         len(serie2[id2]['profile'])])
        d_l = jf.damerau_levenshtein_distance(
            ''.join(serie1[id1]['profile']),
            ''.join(serie2[id2]['profile']))/length

        cicle_time_s1 = (
            serie1[id1]['end_time'] - serie1[id1]['start_time']).total_seconds()
        cicle_time_s2 = (
            serie2[id2]['end_time'] - serie2[id2]['start_time']).total_seconds()
        ae = np.abs(cicle_time_s1 - cicle_time_s2)
        return d_l, ae

# =============================================================================
# mae distance
# =============================================================================

    def mae_metric(self, log_data, simulation_data):
        """
        Calculates the mae metric and

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
        mx_len = len(log_data)
        ae_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
        # Create cost matrix
        # start = timer()
        for i in range(0, mx_len):
            for j in range(0, mx_len):
                cicle_time_s1 = (simulation_data[i]['end_time'] -
                                 simulation_data[i]['start_time']).total_seconds()
                cicle_time_s2 = (log_data[j]['end_time'] -
                                 log_data[j]['start_time']).total_seconds()
                ae = np.abs(cicle_time_s1 - cicle_time_s2)
                ae_matrix[i][j] = ae
        # end = timer()
        # print(end - start)
        ae_matrix = np.array(ae_matrix)
        # Matching using the hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(np.array(ae_matrix))
        # Create response
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx]['caseid'],
                                   sim_order=simulation_data[idx]['profile'],
                                   log_order=log_data[idy]['profile'],
                                   sim_score=(ae_matrix[idx][idy])))
        return similarity
    
# =============================================================================
# Log emd distance
# =============================================================================
    
    def log_emd_metric(self, log_data, simulation_data, criteria='hour'):
        similarity = list()
        window = 1
        # hist_range = [0, int((window * 3600))]
        log_data = pd.DataFrame(log_data)
        simulation_data = pd.DataFrame(simulation_data)

        def split_date_time(dataframe, feature, source):
            day_hour = lambda x: x[feature].hour
            dataframe['hour'] = dataframe.apply(day_hour, axis=1)
            date = lambda x: x[feature].date()
            dataframe['date'] = dataframe.apply(date, axis=1)
            # create time windows
            i = 0
            daily_windows = dict()
            for x in range(24):
                if x % window == 0:
                    i += 1
                daily_windows[x] = i
            dataframe = dataframe.merge(
                pd.DataFrame.from_dict(
                    daily_windows, orient='index').rename_axis('hour'),
                on='hour',
                how='left').rename(columns={0: 'window'})
            dataframe = dataframe[[feature, 'date', 'window']]
            dataframe.rename(columns={feature: 'timestamp'}, inplace=True)
            dataframe['source'] = source
            return dataframe
        data = split_date_time(log_data, 'start_time', 'log')
        data = data.append(
            split_date_time(log_data, 'end_time', 'log'), ignore_index=True)
        data = data.append(
            split_date_time(simulation_data, 'start_time', 'sim'), ignore_index=True)
        data = data.append(
            split_date_time(simulation_data, 'end_time', 'sim'), ignore_index=True)
        data['weekday'] = data.apply(lambda x: x.date.weekday(), axis=1)
        g_criteria = {'hour': 'window', 'day_emd': 'weekday',
                      'day_hour_emd': ['weekday', 'window'], 'cal_emd': 'date'} 
        similarity = list()
        for key, group in data.groupby(g_criteria[criteria]):
            w_df = group.copy()
            w_df = w_df.reset_index()
            basetime = w_df.timestamp.min().floor(freq ='H')
            diftime = lambda x: (x['timestamp'] - basetime).total_seconds()
            w_df['rel_time'] = w_df.apply(diftime, axis=1)
            log_hist = np.histogram(w_df[w_df.source=='log'].rel_time, density=True)
            sim_hist = np.histogram(w_df[w_df.source=='sim'].rel_time, density=True)
            if np.isnan(np.sum(log_hist[0])) or np.isnan(np.sum(sim_hist[0])):
                similarity.append({'window': key,
                                   'sim_score': 0})
            else:
                similarity.append(
                    {'window': key,
                     'sim_score': (1 - wasserstein_distance(log_hist[0],
                                                       sim_hist[0]))})
        return similarity


# =============================================================================
# Support methods
# =============================================================================

    def create_task_alias(self, features):
        """
        Create string alias for tasks names or tuples of tasks-roles names

        Parameters
        ----------
        features : list

        Returns
        -------
        alias : alias dictionary

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
        Scales times values activity based

        Parameters
        ----------
        data : dataframe

        Returns
        -------
        data : dataframe with normalized times

        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        if self.one_timestamp:
            summ = data.groupby(['task'])['duration'].max().to_dict()
            dur_act_norm = (lambda x: x['duration']/summ[x['task']]
                            if summ[x['task']] > 0 else 0)
            df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        else:
            summ = data.groupby(['task'])['processing_time'].max().to_dict()
            proc_act_norm = (lambda x: x['processing_time']/summ[x['task']]
                             if summ[x['task']] > 0 else 0)
            df_modif['proc_act_norm'] = df_modif.apply(proc_act_norm, axis=1)
            # ---
            summ = data.groupby(['task'])['waiting_time'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting_time']/summ[x['task']]
                             if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

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
            [x.update(dict(alias=self.alias[(x[features[0]],
                                             x[features[1]])])) for x in data]
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
# #%%
# with open('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_simo/settings.json') as file:
#     settings = json.load(file)
#     file.close()
    
# data = pd.read_csv('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_simo/process_stats.csv')
# data['end_timestamp'] =  pd.to_datetime(data['end_timestamp'], format=settings['read_options']['timeformat'])
# data['start_timestamp'] =  pd.to_datetime(data['start_timestamp'], format=settings['read_options']['timeformat'])
# evaluation = SimilarityEvaluator(
#     data,
#     settings,
#     0)
#     # metric='tsd')
# evaluation.measure_distance('hour_emd')
# print(evaluation.similarity)
# evaluation.measure_distance('day_emd')
# print(evaluation.similarity)
# evaluation.measure_distance('day_hour_emd')
# print(evaluation.similarity)
# evaluation.measure_distance('cal_emd')
# print(evaluation.similarity)

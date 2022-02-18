"""
Created on Fri Jan 10 11:40:46 2020

@author: Manuel Camargo
"""
import copy
import itertools
import multiprocessing
import string
import time
import traceback
import warnings
from multiprocessing import Pool
from operator import itemgetter

import jellyfish as jf
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from simod.configuration import Configuration, Metric
from . import alpha_oracle as ao
from .alpha_oracle import Rel


class SimilarityEvaluator():
    """
        This class evaluates the similarity of two event-logs
     """

    def __init__(self, log_data: pd.DataFrame, simulation_data: pd.DataFrame, settings: Configuration, max_cases=500,
                 dtype='log'):
        self.dtype = dtype
        self.log_data = copy.deepcopy(log_data)
        self.simulation_data = copy.deepcopy(simulation_data)
        self.max_cases = max_cases
        self.one_timestamp = settings.read_options.one_timestamp
        self._preprocess_data(dtype)

    def _preprocess_data(self, dtype):
        preprocessor = self._get_preprocessor(dtype)
        return preprocessor()

    def _get_preprocessor(self, dtype):
        if dtype == 'log':
            return self._preprocess_log
        elif dtype == 'serie':
            return self._preprocess_serie
        else:
            raise ValueError(dtype)

    def _preprocess_log(self):
        self.ramp_io_perc = 0.2
        self.log_data['source'] = 'log'
        self.simulation_data['source'] = 'simulation'
        data = pd.concat([self.log_data, self.simulation_data], axis=0, ignore_index=True)
        if (('processing_time' not in data.columns) or ('waiting_time' not in data.columns)):
            data = self.calculate_times(data)
        data = self.scaling_data(data)
        # save data
        self.log_data = data[data.source == 'log']
        self.simulation_data = data[data.source == 'simulation']
        self.alias = self.create_task_alias(data, 'task')

        self.alpha_concurrency = ao.AlphaOracle(self.log_data, self.alias, self.one_timestamp, True)
        # reformat and sampling data
        self.log_data = self.reformat_events(self.log_data.to_dict('records'), 'task')
        self.simulation_data = self.reformat_events(self.simulation_data.to_dict('records'), 'task')
        num_traces = int(len(self.simulation_data) * self.ramp_io_perc)
        self.simulation_data = self.simulation_data[num_traces:-num_traces]
        self.log_data = list(map(lambda i: self.log_data[i],
                                 np.random.randint(0, len(self.log_data), len(self.simulation_data))))

    def _preprocess_serie(self):
        # load data
        self.log_data['source'] = 'log'
        self.simulation_data['source'] = 'simulation'

    def measure_distance(self, metric: Metric, verbose=False):
        """
        Measures the distance of two event-logs
        with with tsd or dl and mae distance

        Returns
        -------
        distance : float

        """
        self.verbose = verbose
        # similarity measurement and matching
        evaluator = self._get_evaluator(metric)
        if metric in [Metric.DAY_EMD, Metric.DAY_HOUR_EMD, Metric.CAL_EMD]:
            distance = evaluator(self.log_data, self.simulation_data, criteria=metric)
        else:
            distance = evaluator(self.log_data, self.simulation_data, metric)
        self.similarity = {'metric': metric, 'sim_val': np.mean([x['sim_score'] for x in distance])}

    def _get_evaluator(self, metric: Metric):
        if self.dtype == 'log':
            if metric in [Metric.TSD, Metric.DL, Metric.MAE, Metric.DL_MAE]:
                return self._evaluate_seq_distance
            elif metric is Metric.LOG_MAE:
                return self.log_mae_metric
            elif metric in [Metric.HOUR_EMD, Metric.DAY_EMD, Metric.DAY_HOUR_EMD, Metric.CAL_EMD]:
                return self.log_emd_metric
            else:
                raise ValueError(metric)
        elif self.dtype == 'serie':
            if metric in [Metric.HOUR_EMD, Metric.DAY_EMD, Metric.DAY_HOUR_EMD, Metric.CAL_EMD]:
                return self.serie_emd_metric
            else:
                raise ValueError(metric)
        else:
            raise ValueError(self.dtype)

    # =============================================================================
    # Timed string distance
    # =============================================================================

    def _evaluate_seq_distance(self, log_data, simulation_data, metric: Metric):
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

        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        # define the type of processing sequencial or parallel
        cases = len(set([x['caseid'] for x in log_data]))
        if cases <= self.max_cases:
            args = (metric, simulation_data, log_data,
                    self.alpha_concurrency.oracle,
                    ({'min': 0, 'max': len(simulation_data)},
                     {'min': 0, 'max': len(log_data)}))
            df_matrix = self._compare_traces(args)
        else:
            cpu_count = multiprocessing.cpu_count()
            mx_len = len(log_data)
            ranges = self.define_ranges(mx_len, int(np.ceil(cpu_count / 2)))
            ranges = list(itertools.product(*[ranges, ranges]))
            reps = len(ranges)
            pool = Pool(processes=cpu_count)
            # Generate
            args = [(metric, simulation_data[r[0]['min']:r[0]['max']],
                     log_data[r[1]['min']:r[1]['max']],
                     self.alpha_concurrency.oracle,
                     r) for r in ranges]
            p = pool.map_async(self._compare_traces, args)
            if self.verbose:
                pbar_async(p, f'evaluating {metric}:')
            pool.close()
            # Save results
            df_matrix = pd.concat(list(p.get()), axis=0, ignore_index=True)
        df_matrix.sort_values(by=['i', 'j'], inplace=True)
        df_matrix = df_matrix.reset_index().set_index(['i', 'j'])
        if metric == Metric.DL_MAE:
            dl_matrix = df_matrix[['dl_distance']].unstack().to_numpy()
            mae_matrix = df_matrix[['mae_distance']].unstack().to_numpy()
            # MAE normalized
            max_mae = mae_matrix.max()
            mae_matrix = np.divide(mae_matrix, max_mae)
            # multiple both matrixes by Beta equal to 0.5
            dl_matrix = np.multiply(dl_matrix, 0.5)
            mae_matrix = np.multiply(mae_matrix, 0.5)
            # add each point in between
            cost_matrix = np.add(dl_matrix, mae_matrix)
        else:
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
        # Create response
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx]['caseid'],
                                   sim_order=simulation_data[idx]['profile'],
                                   log_order=log_data[idy]['profile'],
                                   sim_score=(cost_matrix[idx][idy]
                                              if metric == Metric.MAE else
                                              (1 - (cost_matrix[idx][idy])))
                                   )
                              )
        return similarity

    @staticmethod
    def _compare_traces(args):

        def ae_distance(et_1, et_2, st_1, st_2):
            cicle_time_s1 = (et_1 - st_1).total_seconds()
            cicle_time_s2 = (et_2 - st_2).total_seconds()
            ae = np.abs(cicle_time_s1 - cicle_time_s2)
            return ae

        def tsd_alpha(s_1, s_2, p_1, p_2, w_1, w_2, alpha_concurrency):
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

            def calculate_cost(s1_idx, s2_idx):
                t_1 = p_1[s1_idx] + w_1[s1_idx]
                if t_1 > 0:
                    b_1 = (p_1[s1_idx] / t_1)
                    cost = ((b_1 * np.abs(p_2[s2_idx] - p_1[s1_idx])) +
                            ((1 - b_1) * np.abs(w_2[s2_idx] - w_1[s1_idx])))
                else:
                    cost = 0
                return cost

            dist = {}
            lenstr1 = len(s_1)
            lenstr2 = len(s_2)
            for i in range(-1, lenstr1 + 1):
                dist[(i, -1)] = i + 1
            for j in range(-1, lenstr2 + 1):
                dist[(-1, j)] = j + 1
            for i in range(0, lenstr1):
                for j in range(0, lenstr2):
                    if s_1[i] == s_2[j]:
                        cost = calculate_cost(i, j)
                    else:
                        cost = 1
                    dist[(i, j)] = min(
                        dist[(i - 1, j)] + 1,  # deletion
                        dist[(i, j - 1)] + 1,  # insertion
                        dist[(i - 1, j - 1)] + cost  # substitution
                    )
                    if i and j and s_1[i] == s_2[j - 1] and s_1[i - 1] == s_2[j]:
                        if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                            cost = calculate_cost(i, j - 1)
                        dist[(i, j)] = min(dist[(i, j)], dist[i - 2, j - 2] + cost)  # transposition
            return dist[lenstr1 - 1, lenstr2 - 1]

        def gen(metric: Metric, serie1, serie2, oracle, r):
            """Reads the simulation results stats"""
            try:
                df_matrix = list()
                for i, s1_ele in enumerate(serie1):
                    for j, s2_ele in enumerate(serie2):
                        element = {'i': r[0]['min'] + i, 'j': r[1]['min'] + j}
                        if metric in [Metric.TSD, Metric.DL, Metric.DL_MAE]:
                            element['s_1'] = s1_ele['profile']
                            element['s_2'] = s2_ele['profile']
                            element['length'] = max(len(s1_ele['profile']), len(s2_ele['profile']))
                        if metric is Metric.TSD:
                            element['p_1'] = s1_ele['proc_act_norm']
                            element['p_2'] = s2_ele['proc_act_norm']
                            element['w_1'] = s1_ele['wait_act_norm']
                            element['w_2'] = s2_ele['wait_act_norm']
                        if metric in [Metric.MAE, Metric.DL_MAE]:
                            element['et_1'] = s1_ele['end_time']
                            element['et_2'] = s2_ele['end_time']
                            element['st_1'] = s1_ele['start_time']
                            element['st_2'] = s2_ele['start_time']
                        df_matrix.append(element)
                df_matrix = pd.DataFrame(df_matrix)
                if metric is Metric.TSD:
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: tsd_alpha(x.s_1, x.s_2, x.p_1, x.p_2, x.w_1, x.w_2, oracle) / x.length, axis=1)
                elif metric is Metric.DL:
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(''.join(x.s_1), ''.join(x.s_2)) / x.length, axis=1)
                elif metric is Metric.MAE:
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: ae_distance(x.et_1, x.et_2, x.st_1, x.st_2), axis=1)
                elif metric is Metric.DL_MAE:
                    df_matrix['dl_distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(''.join(x.s_1), ''.join(x.s_2)) / x.length, axis=1)
                    df_matrix['mae_distance'] = df_matrix.apply(
                        lambda x: ae_distance(x.et_1, x.et_2, x.st_1, x.st_2), axis=1)
                else:
                    raise ValueError(metric)
                return df_matrix
            except Exception:
                traceback.print_exc()

        return gen(*args)

    # =============================================================================
    # whole log MAE
    # =============================================================================
    def log_mae_metric(self, log_data: list, simulation_data: list, metric: Metric) -> list:
        """
        Measures the MAE distance between two whole logs

        Parameters
        ----------
        log_data : list
        simulation_data : list
        Returns
        -------
        list
        """
        similarity = list()
        log_data = pd.DataFrame(log_data)
        simulation_data = pd.DataFrame(simulation_data)
        log_timelapse = (log_data.end_time.max() - log_data.start_time.min()).total_seconds()
        sim_timelapse = (simulation_data.end_time.max() - simulation_data.start_time.min()).total_seconds()
        similarity.append({'sim_score': np.abs(sim_timelapse - log_timelapse)})
        return similarity

    # =============================================================================
    # Log emd distance
    # =============================================================================

    def log_emd_metric(self, log_data: list, simulation_data: list, criteria: Metric = Metric.HOUR_EMD) -> list:
        """
        Measures the EMD distance between two logs on different aggregation
        levels specified by user by defaul per hour

        Parameters
        ----------
        log_data : list
        simulation_data : list
        criteria : TYPE, optional
            DESCRIPTION. The default is 'hour'.
        Returns
        -------
        list
        """
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
            for hour in range(24):
                if hour % window == 0:
                    i += 1
                daily_windows[hour] = i
            dataframe = dataframe.merge(
                pd.DataFrame.from_dict(daily_windows, orient='index').rename_axis('hour'),
                on='hour',
                how='left').rename(columns={0: 'window'})
            dataframe = dataframe[[feature, 'date', 'window']]
            dataframe.rename(columns={feature: 'timestamp'}, inplace=True)
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], utc=True)
            dataframe['source'] = source
            return dataframe

        data = split_date_time(log_data, 'start_time', 'log')
        data = data.append(split_date_time(log_data, 'end_time', 'log'), ignore_index=True)
        data = data.append(split_date_time(simulation_data, 'start_time', 'sim'), ignore_index=True)
        data = data.append(split_date_time(simulation_data, 'end_time', 'sim'), ignore_index=True)
        data['weekday'] = data.apply(lambda x: x.date.weekday(), axis=1)
        g_criteria = {Metric.HOUR_EMD: 'window', Metric.DAY_EMD: 'weekday', Metric.DAY_HOUR_EMD: ['weekday', 'window'],
                      Metric.CAL_EMD: 'date'}
        similarity = list()
        for key, group in data.groupby(g_criteria[criteria]):
            w_df = group.copy()
            w_df = w_df.reset_index()
            basetime = w_df.timestamp.min().floor(freq='H')
            diftime = lambda x: (x['timestamp'] - basetime).total_seconds()
            w_df['rel_time'] = w_df.apply(diftime, axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                log_hist = np.histogram(w_df[w_df.source == 'log'].rel_time, density=True)
                sim_hist = np.histogram(w_df[w_df.source == 'sim'].rel_time, density=True)
            if np.isnan(np.sum(log_hist[0])) or np.isnan(np.sum(sim_hist[0])):
                similarity.append({'window': key, 'sim_score': 0})
            else:
                similarity.append({'window': key, 'sim_score': wasserstein_distance(log_hist[0], sim_hist[0])})
        return similarity

    # =============================================================================
    # serie emd distance
    # =============================================================================

    def serie_emd_metric(self, log_data, simulation_data, criteria: Metric = Metric.HOUR_EMD):
        similarity = list()
        window = 1
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
                pd.DataFrame.from_dict(daily_windows, orient='index').rename_axis('hour'),
                on='hour', how='left').rename(columns={0: 'window'})
            dataframe = dataframe[[feature, 'date', 'window']]
            dataframe.rename(columns={feature: 'timestamp'}, inplace=True)
            dataframe['source'] = source
            return dataframe

        data = split_date_time(log_data, 'timestamp', 'log')
        data = data.append(split_date_time(simulation_data, 'timestamp', 'sim'), ignore_index=True)
        data['weekday'] = data.apply(lambda x: x.date.weekday(), axis=1)
        g_criteria = {Metric.HOUR_EMD: 'window', Metric.DAY_EMD: 'weekday', Metric.DAY_HOUR_EMD: ['weekday', 'window'],
                      Metric.CAL_EMD: 'date'}
        similarity = list()
        for key, group in data.groupby(g_criteria[criteria]):
            w_df = group.copy()
            w_df = w_df.reset_index()
            basetime = w_df.timestamp.min().floor(freq='H')
            diftime = lambda x: (x['timestamp'] - basetime).total_seconds()
            w_df['rel_time'] = w_df.apply(diftime, axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                log_hist = np.histogram(w_df[w_df.source == 'log'].rel_time, density=True)
                sim_hist = np.histogram(w_df[w_df.source == 'sim'].rel_time, density=True)
            if np.isnan(np.sum(log_hist[0])) or np.isnan(np.sum(sim_hist[0])):
                similarity.append({'window': key, 'sim_score': 1})
            else:
                similarity.append({'window': key, 'sim_score': wasserstein_distance(log_hist[0], sim_hist[0])})
        return similarity

    # =============================================================================
    # Support methods
    # =============================================================================

    def create_task_alias(self, data, features):
        """
        Create string alias for tasks names or tuples of tasks-roles names

        Parameters
        ----------
        features : list

        Returns
        -------
        alias : alias dictionary

        """
        data = data.to_dict('records')
        subsec_set = set()
        if isinstance(features, list):
            task_list = [(x[features[0]], x[features[1]]) for x in data]
        else:
            task_list = [x[features] for x in data]
        [subsec_set.add(x) for x in task_list]
        variables = sorted(list(subsec_set))
        characters = string.ascii_letters + string.digits
        # characters = [chr(i) for i in range(0, len(variables))]
        aliases = list(map(lambda i: characters[i], np.random.randint(0, len(characters), len(variables))))
        alias = dict()
        for i, _ in enumerate(variables):
            alias[variables[i]] = aliases[i]
        return alias

    @staticmethod
    def calculate_times(log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['processing_time'] = 0
        log['multitasking'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: (x['source'], x['caseid']))
        for _, group in itertools.groupby(log, key=lambda x: (x['source'], x['caseid'])):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                dur = (events[i]['end_timestamp'] - events[i]['start_timestamp']).total_seconds()
                if i == 0:
                    wit = 0
                else:
                    wit = (events[i]['start_timestamp'] - events[i - 1]['end_timestamp']).total_seconds()
                events[i]['waiting_time'] = wit if wit >= 0 else 0
                events[i]['processing_time'] = dur
        return pd.DataFrame.from_dict(log)

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
            dur_act_norm = (lambda x: x['duration'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        else:
            summ = data.groupby(['task'])['processing_time'].max().to_dict()
            proc_act_norm = (lambda x: x['processing_time'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['proc_act_norm'] = df_modif.apply(proc_act_norm, axis=1)
            # ---
            summ = data.groupby(['task'])['waiting_time'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting_time'] / summ[x['task']]
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
            columns = ['alias', 'processing_time', 'proc_act_norm', 'waiting_time', 'wait_act_norm']
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
            temp_dict = {**{'caseid': key, 'start_time': trace[0][sort_key], 'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))

    @staticmethod
    def define_ranges(size, num_folds):
        num_events = int(np.round(size / num_folds))
        folds = list()
        for i in range(0, num_folds):
            sidx = i * num_events
            eidx = (i + 1) * num_events
            if i == 0:
                folds.append({'min': 0, 'max': eidx})
            elif i == (num_folds - 1):
                folds.append({'min': sidx, 'max': size})
            else:
                folds.append({'min': sidx, 'max': eidx})
        return folds

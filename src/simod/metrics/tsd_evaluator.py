import itertools
import multiprocessing
import string
from concurrent.futures import ProcessPoolExecutor as Pool
from operator import itemgetter

import jellyfish as jf
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from simod.event_log.column_mapping import EventLogIDs


class TimedStringDistanceEvaluator:
    """Computes DL for two event logs."""

    def __init__(self,
                 log_data: pd.DataFrame,
                 log_ids: EventLogIDs,
                 simulation_data: pd.DataFrame,
                 simulation_log_ids: EventLogIDs,
                 max_cases=500):
        # TODO: avoid deep copying
        self.log_data = log_data.copy(deep=True)
        self.simulation_data = simulation_data.copy(deep=True)

        self.log_ids = log_ids
        self.simulation_log_ids = simulation_log_ids
        self.max_cases = max_cases
        self.similarity = None

        self._preprocess_log()

    def _preprocess_log(self):
        self.ramp_io_perc = 0.2
        self.log_data['source'] = 'log'
        self.simulation_data['source'] = 'simulation'

        # renaming simulation log columns
        renaming_dict = self.simulation_log_ids.renaming_dict(self.log_ids)
        simulation_log_renamed = self.simulation_data.rename(columns=renaming_dict)

        data = pd.concat([self.log_data, simulation_log_renamed], axis=0, ignore_index=True)
        if ('processing_time' not in data.columns) or ('waiting_time' not in data.columns):
            data = self.calculate_times(data)

        data = self.scaling_data(data)

        # save data
        self.log_data = data[data.source == 'log']
        self.simulation_data = data[data.source == 'simulation']
        self.alias = self.create_task_alias(data, self.log_ids.activity)

        # reformat and sampling data
        self.log_data = self.reformat_events(self.log_data.to_dict('records'), self.log_ids.activity)
        self.simulation_data = self.reformat_events(self.simulation_data.to_dict('records'), self.log_ids.activity)
        num_traces = int(len(self.simulation_data) * self.ramp_io_perc)
        self.simulation_data = self.simulation_data[num_traces:-num_traces]
        self.log_data = list(map(lambda i: self.log_data[i],
                                 np.random.randint(0, len(self.log_data), len(self.simulation_data))))

    def measure_distance(self) -> float:
        distance = self._evaluate_seq_distance(self.log_data, self.simulation_data)
        value = np.mean([x['sim_score'] for x in distance])
        return value

    def _evaluate_seq_distance(self, log_data, simulation_data):
        """
        Timed string distance calculation. Returns TSD similarity
        """
        # define the type of processing sequential or parallel
        cases = len(set([x[self.log_ids.case] for x in log_data]))
        if cases <= self.max_cases:
            args = (simulation_data,
                    log_data,
                    ({'min': 0, 'max': len(simulation_data)},
                     {'min': 0, 'max': len(log_data)}))
            df_matrix = self._compare_traces(args)
        else:
            cpu_count = multiprocessing.cpu_count()
            mx_len = len(log_data)
            ranges = self.define_ranges(mx_len, int(np.ceil(cpu_count / 2)))
            ranges = list(itertools.product(*[ranges, ranges]))

            with Pool(max_workers=cpu_count) as pool:
                args = [
                    (simulation_data[r[0]['min']:r[0]['max']],
                     log_data[r[1]['min']:r[1]['max']],
                     r)
                    for r in ranges
                ]
                comparison_results = pool.map(self._compare_traces, args)

            # Save results
            df_matrix = pd.concat(list(comparison_results), axis=0, ignore_index=True)

        df_matrix.sort_values(by=['i', 'j'], inplace=True)
        df_matrix = df_matrix.reset_index().set_index(['i', 'j'])
        cost_matrix = df_matrix[['distance']].unstack().to_numpy()
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))

        # Create response
        similarity = []
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(dict(caseid=simulation_data[idx][self.log_ids.case],
                                   sim_order=simulation_data[idx]['profile'],
                                   log_order=log_data[idy]['profile'],
                                   sim_score=(1 - (cost_matrix[idx][idy]))))
        return similarity

    def _compare_traces(self, args):
        def gen(serie1, serie2, r):
            """Reads the simulation results stats"""
            df_matrix = list()
            for i, s1_ele in enumerate(serie1):
                for j, s2_ele in enumerate(serie2):
                    element = {'i': r[0]['min'] + i, 'j': r[1]['min'] + j}
                    element['s_1'] = s1_ele['profile']
                    element['s_2'] = s2_ele['profile']
                    element['length'] = max(len(s1_ele['profile']), len(s2_ele['profile']))
                    df_matrix.append(element)
            df_matrix = pd.DataFrame(df_matrix)
            df_matrix['distance'] = df_matrix.apply(
                lambda x: jf.damerau_levenshtein_distance(''.join(x.s_1), ''.join(x.s_2)) / x.length, axis=1)
            return df_matrix

        return gen(*args)

    def create_task_alias(self, data, features):
        """Create string alias for tasks names or tuples of tasks-roles names."""
        data = data.to_dict('records')
        subsec_set = set()
        if isinstance(features, list):
            task_list = [(x[features[0]], x[features[1]]) for x in data]
        else:
            task_list = [x[features] for x in data]
        [subsec_set.add(x) for x in task_list]
        variables = sorted(list(subsec_set))
        characters = string.ascii_letters + string.digits
        aliases = list(map(lambda i: characters[i], np.random.randint(0, len(characters), len(variables))))
        alias = dict()
        for i, _ in enumerate(variables):
            alias[variables[i]] = aliases[i]
        return alias

    def calculate_times(self, log):
        """Appends the indexes and relative time to the dataframe."""
        log['processing_time'] = 0
        log['multitasking'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: (x['source'], x[self.log_ids.case]))
        for _, group in itertools.groupby(log, key=lambda x: (x['source'], x[self.log_ids.case])):
            events = list(group)
            events = sorted(events, key=itemgetter(self.log_ids.start_time))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instance there is no previous timestamp
                # to find a range
                dur = (events[i][self.log_ids.end_time] - events[i][self.log_ids.start_time]).total_seconds()
                if i == 0:
                    wit = 0
                else:
                    wit = (events[i][self.log_ids.start_time] - events[i - 1][self.log_ids.end_time]).total_seconds()
                events[i]['waiting_time'] = wit if wit >= 0 else 0
                events[i]['processing_time'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """Scales times values activity based."""
        df_modif = data.copy()
        np.seterr(divide='ignore')
        summ = data.groupby(self.log_ids.activity)['processing_time'].max().to_dict()
        proc_act_norm = (lambda x: x['processing_time'] / summ[x[self.log_ids.activity]]
        if summ[x[self.log_ids.activity]] > 0 else 0)
        df_modif['proc_act_norm'] = df_modif.apply(proc_act_norm, axis=1)
        # ---
        summ = data.groupby(self.log_ids.activity)['waiting_time'].max().to_dict()
        wait_act_norm = (lambda x: x['waiting_time'] / summ[x[self.log_ids.activity]]
        if summ[x[self.log_ids.activity]] > 0 else 0)
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
        sort_key = self.log_ids.start_time
        columns = ['alias', 'processing_time', 'proc_act_norm', 'waiting_time', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x[self.log_ids.case], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x[self.log_ids.case]):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {'profile': serie, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {col: serie, **temp_dict}
            temp_dict = {
                self.log_ids.case: key,
                self.log_ids.start_time: trace[0][sort_key],
                self.log_ids.end_time: trace[-1][sort_key],
                **temp_dict
            }
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter(self.log_ids.start_time))

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

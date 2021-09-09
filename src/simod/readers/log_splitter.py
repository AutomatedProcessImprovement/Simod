# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:35:36 2020

@author: Manuel Camargo
"""
import copy
import itertools
from operator import itemgetter
import numpy as np
import pandas as pd
import random

from memory_profiler import profile


class LogSplitter(object):
    """
    This class reads and parse the elements of a given event-log
    expected format .xes or .csv
    """

    # @profile(stream=open('logs/memprof_LogSplitter.log', 'a+'))
    def __init__(self, log: pd.DataFrame, verbose=True):
        self.log = log
        self._sort_log()

    # @profile(stream=open('logs/memprof_LogSplitter.split_log.log', 'a+'))
    def split_log(self, method: str, size: float, one_timestamp: bool):
        splitter = self._get_splitter(method)
        return splitter(size, one_timestamp)

    def _get_splitter(self, method):
        if method == 'timeline_contained':
            return self._timeline_contained
        elif method == 'timeline_trace':
            return self._timeline_trace
        elif method == 'random':
            return self._random
        else:
            raise ValueError(method)

    def _timeline_contained(self, size: float, one_timestamp: bool) -> None:
        # log = self.log.data.to_dict('records')
        num_events = int(np.round(len(self.log)*(1 - size)))

        df_train = self.log.iloc[num_events:]
        df_test = self.log.iloc[:num_events]

        # Incomplete final traces
        df_train = df_train.sort_values(by=['caseid', 'pos_trace'],
                                        ascending=True)
        inc_traces = pd.DataFrame(df_train.groupby('caseid')
                                  .last()
                                  .reset_index())
        inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
        inc_traces = inc_traces['caseid'].to_list()

        # Drop incomplete traces
        df_test = df_test[~df_test.caseid.isin(inc_traces)]
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])

        df_train = df_train[~df_train.caseid.isin(inc_traces)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        df_test = (df_test
                   .sort_values(key, ascending=True)
                   .reset_index(drop=True).to_dict('records'))
        df_train = (df_train
                    .sort_values(key, ascending=True)
                    .reset_index(drop=True).to_dict('records'))
        return df_train, df_test

    def _timeline_trace(self, size: float, one_timestamp: bool) -> None:
        # log = self.log.data.to_dict('records')
        cases = self.log[self.log.pos_trace == 1]
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        cases = cases.sort_values(key, ascending=False)
        cases = cases.caseid.to_list()
        num_test_cases = int(np.round(len(cases)*(1 - size)))
        test_cases = cases[:num_test_cases]
        train_cases = cases[num_test_cases:]
        df_train = self.log[self.log.caseid.isin(train_cases)]
        df_test = self.log[self.log.caseid.isin(test_cases)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])
        return df_train, df_test

    def _random(self, size: float, one_timestamp: bool) -> None:
        cases = list(self.log.caseid.unique())
        sample_sz = int(np.ceil(len(cases)*size))
        scases = list(map(lambda i: cases[i], np.random.randint(0, len(cases), sample_sz)))
        df_train = self.log[self.log.caseid.isin(scases)]
        df_test = self.log[~self.log.caseid.isin(scases)]
        return df_train, df_test

    # @profile(stream=open('logs/memprof_LogSplitter._sort_log.log', 'a+'))
    def _sort_log(self):
        log = copy.deepcopy(self.log)
        log = sorted(log.to_dict('records'), key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_dict(log)
        log.sort_values(by='end_timestamp', ascending=False, inplace=True)
        self.log = log

import copy
import itertools
from _operator import itemgetter

import numpy as np
import pandas as pd

from simod.event_log.column_mapping import EventLogIDs


class LogSplitter:
    def __init__(self, log: pd.DataFrame, log_ids: EventLogIDs):
        self.log = log
        self.log_ids = log_ids
        self._sort_log()

    def split_log(self, method: str, size: float, one_timestamp: bool = False):
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

    def _timeline_contained(self, size: float, one_timestamp: bool):
        num_events = int(np.round(len(self.log) * (1 - size)))

        df_train = self.log.iloc[num_events:]
        df_test = self.log.iloc[:num_events]

        # Incomplete final traces
        df_train = df_train.sort_values(by=[self.log_ids.case, 'pos_trace'], ascending=True)
        inc_traces = pd.DataFrame(df_train.groupby(self.log_ids.case).last().reset_index())
        inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
        inc_traces = inc_traces[self.log_ids.case].to_list()

        # Drop incomplete traces
        df_test = df_test[~df_test[self.log_ids.case].isin(inc_traces)]
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])

        df_train = df_train[~df_train[self.log_ids.case].isin(inc_traces)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        key = self.log_ids.end_time if one_timestamp else self.log_ids.start_time
        df_test = df_test.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records')
        df_train = df_train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records')
        return df_train, df_test

    def _timeline_trace(self, size: float, one_timestamp: bool):
        cases = self.log[self.log.pos_trace == 1]
        key = self.log_ids.end_time if one_timestamp else self.log_ids.start_time
        cases = cases.sort_values(key, ascending=False)
        cases = cases[self.log_ids.case].to_list()
        num_test_cases = int(np.round(len(cases) * (1 - size)))
        test_cases = cases[:num_test_cases]
        train_cases = cases[num_test_cases:]
        df_train = self.log[self.log[self.log_ids.case].isin(train_cases)]
        df_test = self.log[self.log[self.log_ids.case].isin(test_cases)]
        df_train = df_train.drop(columns=['trace_len', 'pos_trace'])
        df_test = df_test.drop(columns=['trace_len', 'pos_trace'])
        return df_train, df_test

    def _random(self, size: float, one_timestamp: bool):
        cases = list(self.log[self.log_ids.case].unique())
        sample_sz = int(np.ceil(len(cases) * size))
        scases = list(map(lambda i: cases[i], np.random.randint(0, len(cases), sample_sz)))
        df_train = self.log[self.log[self.log_ids.case].isin(scases)]
        df_test = self.log[~self.log[self.log_ids.case].isin(scases)]
        return df_train, df_test

    def _sort_log(self):
        log = copy.deepcopy(self.log)
        log = sorted(log.to_dict('records'), key=lambda x: x[self.log_ids.case])
        for key, group in itertools.groupby(log, key=lambda x: x[self.log_ids.case]):
            events = list(group)
            events = sorted(events, key=itemgetter(self.log_ids.end_time))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_dict(log)
        log.sort_values(by=self.log_ids.end_time, ascending=False, inplace=True)
        self.log = log

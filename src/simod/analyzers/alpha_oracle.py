import itertools
from enum import Enum

from simod.event_log.column_mapping import EventLogIDs


class Rel(Enum):
    FOLLOWS = 1
    PRECEDES = 2
    NOT_CONNECTED = 3
    PARALLEL = 4


class AlphaOracle:
    """This class provides the alpha concurrency oracle information"""

    def __init__(self, log, log_ids: EventLogIDs, tasks_alias, one_timestamp, look_for_loops=False):
        self.log = log
        self.log_ids = log_ids
        self.tasks_alias = tasks_alias
        self.look_for_loops = look_for_loops
        self.oracle = self._discover_concurrency()

    def _discover_concurrency(self):
        seq_flows = self._reformat_events()
        freqs = self._count_freq(seq_flows)
        footprint_matrix = self._create_footprint_matrix()
        self._fill_footprint_matrix(footprint_matrix, freqs)
        if self.look_for_loops:
            for seq in seq_flows:
                for i in range(0, len(seq) - 2, 2):
                    if seq[i] == seq[i + 2]:
                        footprint_matrix[(seq[i], seq[i + 1])] = Rel.PRECEDES
                        footprint_matrix[(seq[i + 1], seq[i])] = Rel.PRECEDES
        return footprint_matrix

    @staticmethod
    def _fill_footprint_matrix(footprint_matrix: dict, freqs: dict):
        for relation in freqs:
            if footprint_matrix[relation] == Rel.NOT_CONNECTED:
                footprint_matrix[relation] = Rel.PRECEDES
                footprint_matrix[(relation[1], relation[0])] = Rel.FOLLOWS
            elif footprint_matrix[relation] == Rel.FOLLOWS:
                footprint_matrix[relation] = Rel.PARALLEL
                footprint_matrix[(relation[1], relation[0])] = Rel.PARALLEL

    def _create_footprint_matrix(self) -> dict:
        footprint_matrix = dict()
        for source in self.tasks_alias.values():
            for target in self.tasks_alias.values():
                footprint_matrix[(source, target)] = Rel.NOT_CONNECTED
        return footprint_matrix

    @staticmethod
    def _count_freq(seq_flows: list) -> dict:
        freqs = dict()
        for flow in seq_flows:
            for i in range(0, len(flow) - 1):
                if (flow[i], flow[i + 1]) in freqs:
                    freqs[(flow[i], flow[i + 1])] += 1
                else:
                    freqs[(flow[i], flow[i + 1])] = 1
        return freqs

    def _reformat_events(self) -> list:
        """Creates series of activities, roles and relative times per trace.
        Params:
            log_df: dataframe.
        Returns:
            list: lists of activities.
        """
        temp_data = list()
        temp_df = self.log.copy()
        temp_df['alias'] = temp_df.apply(lambda x: self.tasks_alias[x[self.log_ids.activity]], axis=1)
        self.log = temp_df
        log_df = self.log.to_dict('records')
        log_df = sorted(log_df, key=lambda x: (x[self.log_ids.case], x[self.log_ids.start_time]))

        for key, group in itertools.groupby(log_df, key=lambda x: x[self.log_ids.case]):
            trace = list(group)
            series = [y['alias'] for y in trace]
            temp_data.append(series)
        return temp_data

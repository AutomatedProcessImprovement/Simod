from operator import itemgetter
from typing import Union

import networkx as nx
import pandas as pd
import scipy
from scipy.stats import pearsonr

from simod.event_log_processing.reader import EventLogReader


class ResourcePoolDiscoverer:
    """Evaluates the tasks durations and associates resources to it."""
    tasks: dict
    users: dict
    roles: list
    resource_table: list
    _data: pd.DataFrame
    _drawing: bool
    _sim_threshold: float
    _activity_key = 'task'
    _resource_key = 'user'

    def __init__(self,
                 log: Union[EventLogReader, pd.DataFrame],
                 drawing=False,
                 sim_threshold=0.7,
                 activity_key='task',
                 resource_key='user'):
        self._activity_key = activity_key
        self._resource_key = resource_key
        self._drawing = drawing
        self._sim_threshold = sim_threshold
        self._data = self.__read_resource_pool(log)
        self.tasks = {val: i for i, val in enumerate(self._data[self._activity_key].unique())}
        self.users = {val: i for i, val in enumerate(self._data[self._resource_key].unique())}
        self.roles, self.resource_table = self.__discover_roles()

    def __read_resource_pool(self, log: Union[EventLogReader, pd.DataFrame]):
        if isinstance(log, EventLogReader):
            filtered_list = pd.DataFrame(log.data)[[self._activity_key, self._resource_key]]
        elif isinstance(log, pd.DataFrame):
            filtered_list = log[[self._activity_key, self._resource_key]]
        else:
            raise TypeError('Log must be LogReader or pd.DataFrame')
        filtered_list = filtered_list[~filtered_list[self._activity_key].isin(['Start', 'End'])]
        filtered_list = filtered_list[filtered_list[self._resource_key] != 'AUTO']
        return filtered_list

    def __discover_roles(self):
        associations = lambda x: (self.tasks[x[self._activity_key]], self.users[x[self._resource_key]])
        self._data['ac_rl'] = self._data.apply(associations, axis=1)

        freq_matrix = self._data.groupby(by='ac_rl')[self._activity_key].count().reset_index().rename(
            columns={self._activity_key: 'freq'})
        freq_matrix = {x['ac_rl']: x['freq'] for x in freq_matrix.to_dict('records')}

        profiles = self.__build_profile(freq_matrix)

        # NOTE: Pearson coefficient calculation might fail if too few resources
        try:
            # building of a correl matrix between resources profiles
            correl_matrix = self.__det_correl_matrix(profiles)
            # creation of a rel network between resources
            g = nx.Graph()
            for user in self.users.values():
                g.add_node(user)
            for rel in correl_matrix:
                # creation of edges between nodes excluding the same elements
                # and those below the similarity threshold
                if rel['distance'] > self._sim_threshold and rel['x'] != rel['y']:
                    g.add_edge(rel['x'],
                               rel['y'],
                               weight=rel['distance'])
            # extraction of fully connected subgraphs as roles
            sub_graphs = list((g.subgraph(c) for c in nx.connected_components(g)))
            # role definition from graph
            roles = self.__role_definition(sub_graphs)
            return roles
        except ValueError:
            members = self._data[self._resource_key].unique()
            quantity = len(members)
            role = 'Role 1'
            roles = [{'role': role, 'quantity': quantity, 'members': members}]
            resource_table = [{'role': role, 'resource': member} for member in members]
            return roles, resource_table

    def __build_profile(self, freq_matrix):
        profiles = list()
        for user, idx in self.users.items():
            profile = [0, ] * len(self.tasks)
            for ac_rl, freq in freq_matrix.items():
                if idx == ac_rl[1]:
                    profile[ac_rl[0]] = freq
            profiles.append({self._resource_key: idx, 'profile': profile})
        return profiles

    def __det_correl_matrix(self, profiles):
        correl_matrix = list()
        for profile_x in profiles:
            for profile_y in profiles:
                x = scipy.array(profile_x['profile'])
                y = scipy.array(profile_y['profile'])
                r_row, p_value = pearsonr(x, y)
                correl_matrix.append(({'x': profile_x[self._resource_key],
                                       'y': profile_y[self._resource_key],
                                       'distance': r_row}))
        return correl_matrix

    def __role_definition(self, sub_graphs):
        user_index = {v: k for k, v in self.users.items()}
        records = list()
        for i in range(0, len(sub_graphs)):
            users_names = [user_index[x] for x in sub_graphs[i]]
            records.append({'role': 'Role ' + str(i + 1),
                            'quantity': len(sub_graphs[i]),
                            'members': users_names})
        # Sort roles by number of resources
        records = sorted(records, key=itemgetter('quantity'), reverse=True)
        for i in range(0, len(records)):
            records[i]['role'] = 'Role ' + str(i + 1)
        resource_table = list()
        for record in records:
            for member in record['members']:
                resource_table.append({'role': record['role'],
                                       'resource': member})
        return records, resource_table

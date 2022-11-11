import networkx as nx
import pandas as pd
from networkx import DiGraph
from tqdm import tqdm

from simod import utilities as sup
from simod.configuration import PDFMethod
from simod.discovery.distribution import get_best_distribution
from simod.event_log.column_mapping import EventLogIDs


class TaskEvaluator:
    """This class evaluates the tasks durations and associates resources to it."""
    elements_data: pd.DataFrame
    pdef_method: PDFMethod

    _log: pd.DataFrame
    _log_ids: EventLogIDs
    _resource_pool_metadata: dict
    _model_data: pd.DataFrame
    _tasks: list

    def __init__(
            self,
            process_graph: DiGraph,
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            resource_pool: dict,
            pdef_method: PDFMethod):
        self._tasks = self._get_task_list(process_graph)
        self._model_data = self._get_model_data(process_graph)
        self._log_ids = log_ids

        # calculating processing time
        time_delta = log[log_ids.end_time] - log[log_ids.start_time]
        time_delta_in_seconds = list(map(lambda x: x.total_seconds(), time_delta))
        log[log_ids.processing_time] = time_delta_in_seconds
        self._log = log

        self._resource_pool_metadata = resource_pool

        self.pdef_method = pdef_method

        self.elements_data = self._evaluate_tasks()

    def _evaluate_tasks(self) -> pd.DataFrame:
        """Process the task data and association of resources. """
        # processing time discovery method
        if self.pdef_method is PDFMethod.AUTOMATIC:
            elements_data = self._mine_processing_time()
        # elif self.pdef_method == PDFMethod.DEFAULT:
        #     elements_data = self._default_processing_time()
        else:
            raise ValueError(self.pdef_method)

        # Resource association
        elements_data = self._associate_resource(elements_data)
        elements_data = elements_data.to_dict('records')
        elements_data = self._add_start_end_info(elements_data)

        return elements_data

    # def _default_processing_time(self):
    #     elements_data = self._default_values()
    #     elements_data = pd.DataFrame(elements_data)
    #     return elements_data

    def _mine_processing_time(self):
        """Performs the mining of activities durations from data."""
        elements_data = list()
        for task in tqdm(self._tasks, desc='mining tasks distributions:'):
            s_key = self._log_ids.processing_time
            task_processing = self._log[self._log[self._log_ids.activity] == task][s_key].tolist()

            dist = get_best_distribution(task_processing)

            elements_data.append({'id': sup.gen_id(),
                                  'name': task,
                                  'distribution': dist})
        elements_data = pd.DataFrame(elements_data)
        elements_data = elements_data.merge(
            self._model_data[['name', 'elementid']], on='name', how='left')
        return elements_data

    # def _default_values(self):
    #     """Performs the mining of activities durations from data."""
    #     elements_data = list()
    #     for task in tqdm(self._tasks, desc='mining tasks distributions:'):
    #         s_key = self._log_ids.processing_time
    #         task_processing = self._log[self._log[self._log_ids.activity] == task][s_key].tolist()
    #         try:
    #             mean_time = np.mean(task_processing) if task_processing else 0
    #         except:
    #             mean_time = 0
    #         elements_data.append({'id': sup.gen_id(),
    #                               'type': 'EXPONENTIAL',
    #                               'name': task,
    #                               'mean': str(0),
    #                               'arg1': str(np.round(mean_time, 2)),
    #                               'arg2': str(0)})
    #     elements_data = pd.DataFrame(elements_data)
    #     elements_data = elements_data.merge(self._task_ids[['name', 'elementid']], on='name', how='left')
    #     return elements_data.to_dict('records')

    def _add_start_end_info(self, elements_data):
        # records creation
        temp_elements_data = list()
        default_record = {'distribution_name': 'FIXED', 'distribution_params': [{'value': 0}]}
        for task in ['start', 'end']:
            temp_elements_data.append({**{'id': sup.gen_id(), 'name': task}, **default_record})
        temp_elements_data = pd.DataFrame(temp_elements_data)

        temp_elements_data = temp_elements_data.merge(
            self._model_data[['name', 'elementid']],
            on='name',
            how='left').sort_values(by='name')
        temp_elements_data['r_name'] = 'SYSTEM'

        # resource id addition
        resource_id = pd.DataFrame([self._resource_pool_metadata])[['id', 'name']] \
            .rename(columns={'id': 'resource', 'name': 'r_name'})

        temp_elements_data = temp_elements_data \
            .merge(resource_id, on='r_name', how='left') \
            .drop(columns=['r_name'])

        # Appending to the elements data
        temp_elements_data = temp_elements_data.to_dict('records')
        elements_data.extend(temp_elements_data)
        return elements_data

    def _associate_resource(self, elements_data):
        """Merge the resource information with the task data."""
        roles_table = self._log[[self._log_ids.case, 'role', self._log_ids.activity]] \
            .groupby([self._log_ids.activity, 'role']) \
            .count() \
            .sort_values(by=[self._log_ids.case]) \
            .groupby(level=0) \
            .tail(1) \
            .reset_index()

        resource_id = pd.DataFrame([self._resource_pool_metadata])[['id', 'name']] \
            .rename(columns={'id': 'resource', 'name': 'resource_name'})

        roles_table = roles_table \
            .merge(resource_id, left_on='role', right_on='resource_name', how='left') \
            .drop(columns=['role', 'resource_name', self._log_ids.case])

        elements_data = elements_data \
            .merge(roles_table, left_on='name', right_on=self._log_ids.activity, how='left') \
            .drop(columns=[self._log_ids.activity])

        elements_data['resource'].fillna(self._resource_pool_metadata['name'], inplace=True)

        return elements_data

    @staticmethod
    def _get_task_list(process_graph):
        """Extracts the tasks list from the BPM model."""
        tasks = list(filter(lambda x: process_graph.nodes[x]['type'] == 'task', list(nx.nodes(process_graph))))
        tasks = [process_graph.nodes[x]['name'] for x in tasks]
        return tasks

    @staticmethod
    def _get_model_data(process_graph):
        """Extracts the tasks data from the BPM model."""
        model_data = pd.DataFrame.from_dict(dict(process_graph.nodes.data()), orient='index')
        model_data = model_data[model_data.type.isin(['task', 'start', 'end'])].rename(columns={'id': 'elementid'})
        return model_data

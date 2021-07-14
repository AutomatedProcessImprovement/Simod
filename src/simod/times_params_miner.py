import itertools

import pandas as pd
import utils.support as sup

from .configuration import Configuration, PDFMethod, CalculationMethod
from .decorators import safe_exec
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.role_discovery import ResourcePoolAnalyser
from .extraction.schedule_tables import TimeTablesCreator
from .extraction.tasks_evaluator import TaskEvaluator


class TimesParametersMiner():
    """
    This class extracts all the BPS parameters
    """

    def __init__(self, log, bpmn, process_graph, conformant_traces, process_stats, settings: Configuration):
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        self.process_stats = process_stats
        self.conformant_traces = conformant_traces
        # inter-arrival times and durations by default mean an exponential
        # 'manual', 'automatic', 'semi-automatic', 'default'
        self.settings.pdef_method = PDFMethod.AUTOMATIC
        self.parameters = dict()
        self.resource_table = pd.DataFrame()
        self.is_safe = True

    def extract_parameters(self, num_inst, start_time) -> None:
        """
        main method for parameters extraction
        """
        self.is_safe = self._mine_resources(is_safe=self.is_safe)
        self.is_safe = self._mine_interarrival(is_safe=self.is_safe)
        self.is_safe = self._process_tasks(is_safe=self.is_safe)
        # TODO: Num of test partition
        self.parameters['instances'] = num_inst
        self.parameters['start_time'] = start_time

    @safe_exec
    def _mine_resources(self, **kwargs) -> None:
        """
        Analysing resource pool LV917 or 247
        """
        res_analyzer = ResourcePoolAnalyser(self.log, sim_threshold=self.settings.rp_similarity)
        ttcreator = TimeTablesCreator(self.settings)
        args = {'res_cal_met': self.settings.res_cal_met,
                'arr_cal_met': self.settings.arr_cal_met,
                'resource_table': res_analyzer.resource_table}

        if not isinstance(args['res_cal_met'], CalculationMethod):
            args['res_cal_met'] = CalculationMethod.from_str(args['res_cal_met'])
        if not isinstance(args['arr_cal_met'], CalculationMethod):
            args['arr_cal_met'] = CalculationMethod.from_str(args['arr_cal_met'])

        ttcreator.create_timetables(args)
        resource_pool = self._create_resource_pool(res_analyzer.resource_table,
                                                   ttcreator.res_ttable_name)
        self.parameters['resource_pool'] = resource_pool
        self.parameters['time_table'] = ttcreator.time_table
        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        self.process_stats = self.process_stats.merge(resource_table, on='resource', how='left')
        self.resource_table = resource_table

    @safe_exec
    def _mine_interarrival(self, **kwargs) -> None:
        """
        Calculates the inter-arrival rate
        """
        inter_evaluator = InterArrivalEvaluator(self.process_graph, self.conformant_traces, self.settings)
        self.parameters['arrival_rate'] = inter_evaluator.dist

    @safe_exec
    def _process_tasks(self, **kwargs) -> None:
        """
        Tasks id information
        """
        tevaluator = TaskEvaluator(self.process_graph, self.process_stats, self.parameters['resource_pool'],
                                   self.settings)
        self.parameters['elements_data'] = tevaluator.elements_data

    @staticmethod
    def _create_resource_pool(resource_table, table_name) -> list():
        """
        Creates resource pools and associate them the default timetable
        in BIMP format
        """
        resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM',
                          'total_amount': '20', 'costxhour': '20',
                          'timetable_id': table_name['arrival']}]
        data = sorted(resource_table, key=lambda x: x['role'])
        for key, group in itertools.groupby(data, key=lambda x: x['role']):
            res_group = [x['resource'] for x in list(group)]
            r_pool_size = str(len(res_group))
            name = (table_name['resources'] if 'resources' in table_name.keys()
                    else table_name[key])
            resource_pool.append({'id': sup.gen_id(),
                                  'name': key,
                                  'total_amount': r_pool_size,
                                  'costxhour': '20',
                                  'timetable_id': name})
        return resource_pool

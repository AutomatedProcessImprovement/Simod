import itertools
import types
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, List

import pandas as pd
from networkx import DiGraph
from simod.structure_optimizer import mine_resources
from utils import support as sup

from .cli_formatter import print_step
from .configuration import Configuration, CalculationMethod
from .extraction.gateways_probabilities import GatewaysEvaluator
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.log_replayer import LogReplayer
from .extraction.role_discovery import ResourcePoolAnalyser
from .extraction.schedule_tables import TimeTablesCreator
from .extraction.tasks_evaluator import TaskEvaluator
from .readers.bpmn_reader import BpmnReader


class Operator(Protocol):
    input: Any
    output: Any

    @abstractmethod
    def __init__(self, input: Any, output: Any):
        self.input = input
        self.output = output
        self._execute()

    @abstractmethod
    def _execute(self):
        raise NotImplementedError


class Pipeline:
    input: dict = {}
    pipeline: list = []
    output: Any

    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output

    def set_pipeline(self, operators: List[Operator] = None):
        self.pipeline = operators

    def execute(self):
        if not self.pipeline:
            raise ValueError('Pipeline.pipeline is not specified')

        for operator in self.pipeline:
            operator(input=self.input, output=self.output)


########################################################################################################################
# Implementations
########################################################################################################################


# General
########################################################################################################################

@dataclass
class ParameterExtractionInput:
    log_traces: list = None
    bpmn: BpmnReader = None
    process_graph: DiGraph = None
    settings: Configuration = None


@dataclass
class ParameterExtractionOutput:
    process_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    resource_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    conformant_traces: list = field(default_factory=list)
    resource_pool: list = field(default_factory=list)
    time_table: List[str] = field(default_factory=list)
    arrival_rate: dict = field(default_factory=dict)
    sequences: list = field(default_factory=list)
    elements_data: list = field(default_factory=list)


class InterArrivalMiner(Operator):
    input: ParameterExtractionInput
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInput, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Inter-arrival Miner')
        inter_evaluator = InterArrivalEvaluator(
            self.input.process_graph, self.output.conformant_traces, self.input.settings)
        self.output.arrival_rate = inter_evaluator.dist


class GatewayProbabilitiesMiner(Operator):
    input: ParameterExtractionInput
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInput, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Gateway Probabilities Miner')
        evaluator = GatewaysEvaluator(self.input.process_graph, self.input.settings.gate_management)
        sequences = evaluator.probabilities
        for seq in sequences:
            seq['elementid'] = self.input.bpmn.find_sequence_id(seq['gatewayid'], seq['out_path_id'])
        self.output.sequences = sequences


class TasksProcessor(Operator):
    input: ParameterExtractionInput
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInput, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Tasks Processor')
        evaluator = TaskEvaluator(
            self.input.process_graph, self.output.process_stats, self.output.resource_pool, self.input.settings)
        self.output.elements_data = evaluator.elements_data


# For Discoverer
########################################################################################################################

@dataclass
class ParameterExtractionInputForDiscoverer:
    log: types.SimpleNamespace
    bpmn: BpmnReader = None
    process_graph: DiGraph = None
    settings: Configuration = None


class LogReplayerForDiscoverer(Operator):
    input: ParameterExtractionInputForDiscoverer
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInputForDiscoverer, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Log Replayer')
        replayer = LogReplayer(self.input.process_graph, self.input.log.get_traces(), self.input.settings,
                               msg='reading conformant training traces')
        self.output.process_stats = replayer.process_stats
        self.output.conformant_traces = replayer.conformant_traces


class ResourceMinerForDiscoverer(Operator):
    input: ParameterExtractionInputForDiscoverer
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInputForDiscoverer, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        """Analysing resource pool LV917 or 247"""

        def create_resource_pool(resource_table, table_name) -> list:
            """Creates resource pools and associate them the default timetable in BIMP format"""
            resource_pool = [{'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM', 'total_amount': '20', 'costxhour': '20',
                              'timetable_id': table_name['arrival']}]
            data = sorted(resource_table, key=lambda x: x['role'])
            for key, group in itertools.groupby(data, key=lambda x: x['role']):
                res_group = [x['resource'] for x in list(group)]
                r_pool_size = str(len(res_group))
                name = (table_name['resources'] if 'resources' in table_name.keys() else table_name[key])
                resource_pool.append(
                    {'id': sup.gen_id(), 'name': key, 'total_amount': r_pool_size, 'costxhour': '20',
                     'timetable_id': name})
            return resource_pool

        print_step('Resource Miner')

        res_analyzer = ResourcePoolAnalyser(self.input.log, sim_threshold=self.input.settings.rp_similarity)
        ttcreator = TimeTablesCreator(self.input.settings)
        args = {'res_cal_met': self.input.settings.res_cal_met,
                'arr_cal_met': self.input.settings.arr_cal_met,
                'resource_table': res_analyzer.resource_table}

        if not isinstance(args['res_cal_met'], CalculationMethod):
            args['res_cal_met'] = self.input.settings.res_cal_met
        if not isinstance(args['arr_cal_met'], CalculationMethod):
            args['arr_cal_met'] = self.input.settings.arr_cal_met

        ttcreator.create_timetables(args)
        resource_pool = create_resource_pool(res_analyzer.resource_table, ttcreator.res_ttable_name)
        self.output.resource_pool = resource_pool
        self.output.time_table = ttcreator.time_table

        # Adding role to process stats
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        self.output.process_stats = self.output.process_stats.merge(resource_table, on='resource', how='left')
        self.output.resource_table = resource_table


# For Structure Optimizer
########################################################################################################################

class LogReplayerForStructureOptimizer(Operator):
    input: ParameterExtractionInput
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInput, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        print_step('Log Replayer')
        replayer = LogReplayer(self.input.process_graph, self.input.log_traces, self.input.settings,
                               msg='reading conformant training traces')
        self.output.process_stats = replayer.process_stats
        self.output.conformant_traces = replayer.conformant_traces


class ResourceMinerForStructureOptimizer(Operator):
    input: ParameterExtractionInput
    output: ParameterExtractionOutput

    def __init__(self, input: ParameterExtractionInput, output: ParameterExtractionOutput):
        self.input = input
        self.output = output
        self._execute()

    def _execute(self):
        """Analysing resource pool LV917 or 247"""
        print_step('Resource Miner')
        parameters = mine_resources(self.input.settings)
        self.output.resource_pool = parameters['resource_pool']
        self.output.time_table = parameters['time_table']

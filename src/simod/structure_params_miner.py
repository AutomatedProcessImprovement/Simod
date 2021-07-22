from dataclasses import dataclass, field
from typing import List

import pandas as pd
from networkx import DiGraph
from simod.cli_formatter import print_step
from simod.parameter_extraction import Operator
from simod.readers.bpmn_reader import BpmnReader

from .configuration import Configuration, CalculationMethod, DataType
from .extraction.gateways_probabilities import GatewaysEvaluator
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.log_replayer import LogReplayer
from .extraction.schedule_tables import TimeTablesCreator
from .extraction.tasks_evaluator import TaskEvaluator


@dataclass
class ParameterExtractionInput:
    # log: LogReader
    log_traces: list = None
    bpmn: BpmnReader = None
    process_graph: DiGraph = None
    settings: Configuration = None
    # rp_similarity: float


@dataclass
class ParameterExtractionOutput:
    process_stats: list = field(default_factory=list)
    resource_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    conformant_traces: list = field(default_factory=list)
    resource_pool: list = field(default_factory=list)
    time_table: List[str] = field(default_factory=list)
    arrival_rate: dict = field(default_factory=dict)
    sequences: list = field(default_factory=list)
    elements_data: list = field(default_factory=list)


class LogReplayerForStructureOptimizerPipeline(Operator):
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
        self.output.process_stats['role'] = 'SYSTEM'  # TODO: what is this for?


class ResourceMinerForStructureOptimizerPipeline(Operator):
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


class InterArrivalMinerForStructureOptimizerPipeline(Operator):
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


class GatewayProbabilitiesMinerForStructureOptimizerPipeline(Operator):
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


class TasksProcessorForStructureOptimizerPipeline(Operator):
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


def mine_resources(settings: Configuration):
    parameters = dict()
    settings.res_cal_met = CalculationMethod.DEFAULT
    settings.res_dtype = DataType.DT247
    settings.arr_cal_met = CalculationMethod.DEFAULT
    settings.arr_dtype = DataType.DT247
    time_table_creator = TimeTablesCreator(settings)
    args = {'res_cal_met': settings.res_cal_met, 'arr_cal_met': settings.arr_cal_met}

    if not isinstance(args['res_cal_met'], CalculationMethod):
        args['res_cal_met'] = CalculationMethod.from_str(args['res_cal_met'])
    if not isinstance(args['arr_cal_met'], CalculationMethod):
        args['arr_cal_met'] = CalculationMethod.from_str(args['arr_cal_met'])

    time_table_creator.create_timetables(args)
    resource_pool = [
        {'id': 'QBP_DEFAULT_RESOURCE', 'name': 'SYSTEM', 'total_amount': '100000', 'costxhour': '20',
         'timetable_id': time_table_creator.res_ttable_name['arrival']}
    ]

    parameters['resource_pool'] = resource_pool
    parameters['time_table'] = time_table_creator.time_table
    return parameters

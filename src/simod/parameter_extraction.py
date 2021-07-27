from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, List

import pandas as pd
from networkx import DiGraph

from .cli_formatter import print_step
from .configuration import Configuration
from .extraction.gateways_probabilities import GatewaysEvaluator
from .extraction.interarrival_definition import InterArrivalEvaluator
from .extraction.tasks_evaluator import TaskEvaluator
from .readers.bpmn_reader import BpmnReader


# Parameters Extraction Interface

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
    operators: list = []
    output: Any

    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output

    def set_pipeline(self, operators: List[Operator] = None):
        self.operators = operators

    def execute(self):
        if not self.operators:
            raise ValueError('Pipeline.pipeline is not specified')

        for operator in self.operators:
            operator(input=self.input, output=self.output)


# Parameters Extraction Implementations: General

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

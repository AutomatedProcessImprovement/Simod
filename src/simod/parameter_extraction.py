from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from networkx import DiGraph

from .configuration import Configuration, GateManagement
from .readers.bpmn_reader import BpmnReader
from .readers.log_reader import LogReader


class GenericTasksProcessor(ABC):
    elements_data: list

    @abstractmethod
    def __init__(self, process_graph: DiGraph, process_stats: list, resource_pool: list, settings: Configuration):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)


class GenericGatewaysProbabilitiesMiner(ABC):
    sequences: list

    @abstractmethod
    def __init__(self, process_graph: DiGraph, bpmn: BpmnReader, gate_management: GateManagement):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)


class GenericInterArrivalMiner(ABC):
    arrival_rate: dict

    @abstractmethod
    def __init__(self, process_graph: DiGraph, conformant_traces: list, settings: Configuration):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)


class GenericLogReplayer(ABC):
    process_stats: dict
    conformant_traces: list

    @abstractmethod
    def __init__(self, process_graph: DiGraph, log_traces: list, settings: Configuration):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)


class GenericResourceMiner(ABC):
    resource_pool: list
    time_table: Any

    @abstractmethod
    def __init__(self, log: LogReader, rp_similarity: float, settings: Configuration):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)


@dataclass
class GenericParameterExtractionOutput:
    parameters: dict = field(default_factory=dict)
    process_stats: list = field(default_factory=list)
    resource_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    conformant_traces: list = field(default_factory=list)


class GenericParameterExtraction:
    # initial parameters
    log: LogReader
    bpmn: BpmnReader
    process_graph: DiGraph
    settings: Configuration

    # pipeline classes
    log_replayer: GenericLogReplayer = None
    resource_miner: GenericResourceMiner = None
    inter_arrival_miner: GenericInterArrivalMiner = None
    gateway_probabilities_miner: GenericGatewaysProbabilitiesMiner = None
    tasks_processor: GenericTasksProcessor = None

    # output data
    output: GenericParameterExtractionOutput

    def __init__(self, log, bpmn, process_graph, settings: Configuration):
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        self.output = GenericParameterExtractionOutput()

    def setup_pipeline(self,
                       log_replayer=None,
                       resource_miner=None,
                       inter_arrival_miner=None,
                       gateway_probabilities_miner=None,
                       tasks_processor=None):
        self.log_replayer = log_replayer
        self.resource_miner = resource_miner
        self.inter_arrival_miner = inter_arrival_miner
        self.gateway_probabilities_miner = gateway_probabilities_miner
        self.tasks_processor = tasks_processor

    def set_output_parameters(self, **kwargs):
        self.output.parameters.update(kwargs)

    def execute(self):
        if self.log_replayer:
            replayer = self.log_replayer(self.process_graph, self.log.get_traces(), self.settings)
            self.output.conformant_traces = replayer.conformant_traces
            self.output.process_stats = replayer.process_stats

        if self.resource_miner:
            resource_miner = self.resource_miner(self.log, self.settings.rp_similarity, self.settings)
            self.output.parameters['resource_pool'] = resource_miner.resource_pool
            self.output.parameters['time_table'] = resource_miner.time_table

        if self.inter_arrival_miner:
            inter_arrival_miner = self.inter_arrival_miner(
                self.process_graph, self.output.conformant_traces, self.settings)
            self.output.parameters['arrival_rate'] = inter_arrival_miner.arrival_rate

        if self.gateway_probabilities_miner:
            gateway_miner = self.gateway_probabilities_miner(
                self.process_graph, self.bpmn, self.settings.gate_management)
            self.output.parameters['sequences'] = gateway_miner.sequences

        if self.tasks_processor:
            processor = self.tasks_processor(
                self.process_graph, self.output.process_stats, self.output.parameters['resource_pool'], self.settings)
            self.output.parameters['elements_data'] = processor.elements_data

        return self.output




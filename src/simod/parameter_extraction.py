from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, List

import pandas as pd
from networkx import DiGraph

from .configuration import Configuration
from .readers.bpmn_reader import BpmnReader


@dataclass
class ParameterExtractionInput:
    # log: LogReader
    log_traces: list
    bpmn: BpmnReader
    process_graph: DiGraph
    settings: Configuration
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

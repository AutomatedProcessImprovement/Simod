from abc import abstractmethod
from typing import Any, Protocol, List


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

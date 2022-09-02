import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import networkx as nx

from simod.configuration import BPMN_NAMESPACE_URI
from simod.readers import bpm_graph


class BPMNReader:
    """BPMN 2.0 model reader."""
    model_path: Path

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.tree = ET.parse(model_path)
        self.root = self.tree.getroot()
        self.ns = {'xmlns': BPMN_NAMESPACE_URI}

    def as_graph(self) -> nx.DiGraph:
        return bpm_graph.from_bpmn_reader(self)

    def read_activities(self):
        """Activities information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for task in process.findall('xmlns:task', self.ns):
                task_id = task.get('id')
                task_name = task.get('name')
                values.append(dict(task_id=task_id, task_name=task_name))
        return values

    def read_exclusive_gateways(self):
        """Exclusive gateways information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for ex_gateway in process.findall('xmlns:exclusiveGateway', self.ns):
                gate_id = ex_gateway.get('id')
                gate_name = ex_gateway.get('name')
                gate_dir = ex_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_inclusive_gateways(self):
        """Inclusive gateways information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for inc_gateway in process.findall('xmlns:inclusiveGateway', self.ns):
                gate_id = inc_gateway.get('id')
                gate_name = inc_gateway.get('name')
                gate_dir = inc_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_parallel_gateways(self):
        """Parallel gateways information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for para_gateway in process.findall('xmlns:parallelGateway', self.ns):
                gate_id = para_gateway.get('id')
                gate_name = para_gateway.get('name')
                gate_dir = para_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_start_events(self):
        """Start events information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for start_event in process.findall('xmlns:startEvent', self.ns):
                start_id = start_event.get('id')
                start_name = start_event.get('name')
                values.append(dict(start_id=start_id, start_name=start_name))
        return values

    def read_end_events(self):
        """End events information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for end_event in process.findall('xmlns:endEvent', self.ns):
                end_id = end_event.get('id')
                end_name = end_event.get('name')
                values.append(dict(end_id=end_id, end_name=end_name))
        return values

    def read_intermediate_catch_events(self):
        """Intermediate catch events information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for timer_event in process.findall('xmlns:intermediateCatchEvent', self.ns):
                timer_id = timer_event.get('id')
                timer_name = timer_event.get('name')
                values.append(dict(timer_id=timer_id, timer_name=timer_name))
        return values

    def read_sequence_flows(self):
        """Sequence flows information from the model."""
        values = []
        for process in self.root.findall('xmlns:process', self.ns):
            for sequence in process.findall('xmlns:sequenceFlow', self.ns):
                sf_id = sequence.get('id')
                source = sequence.get('sourceRef')
                target = sequence.get('targetRef')
                values.append(dict(sf_id=sf_id, source=source, target=target))
        return values

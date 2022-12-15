import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import lxml.etree
import networkx as nx
import xmltodict as xtd

from simod.bpm import graph
from simod.configuration import BPMN_NAMESPACE_URI, QBP_NAMESPACE_URI


class BPMNReaderWriter:
    """
    BPMN 2.0 model reader and writer.
    """

    model_path: Path

    tree: lxml.etree.ElementTree
    namespace: dict
    _root: lxml.etree.Element

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.tree = lxml.etree.parse(str(self.model_path))
        self._root = self.tree.getroot()
        self.namespace = {'xmlns': BPMN_NAMESPACE_URI}

    def as_graph(self) -> nx.DiGraph:
        return graph.from_bpmn_reader(self)

    def read_activities(self):
        """Activities information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for task in process.findall('xmlns:task', self.namespace):
                task_id = task.get('id')
                task_name = task.get('name')
                values.append(dict(task_id=task_id, task_name=task_name))
        return values

    def read_exclusive_gateways(self):
        """Exclusive gateways information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for ex_gateway in process.findall('xmlns:exclusiveGateway', self.namespace):
                gate_id = ex_gateway.get('id')
                gate_name = ex_gateway.get('name')
                gate_dir = ex_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_inclusive_gateways(self):
        """Inclusive gateways information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for inc_gateway in process.findall('xmlns:inclusiveGateway', self.namespace):
                gate_id = inc_gateway.get('id')
                gate_name = inc_gateway.get('name')
                gate_dir = inc_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_parallel_gateways(self):
        """Parallel gateways information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for para_gateway in process.findall('xmlns:parallelGateway', self.namespace):
                gate_id = para_gateway.get('id')
                gate_name = para_gateway.get('name')
                gate_dir = para_gateway.get('gatewayDirection')
                values.append(dict(gate_id=gate_id, gate_name=gate_name, gate_dir=gate_dir))
        return values

    def read_start_events(self):
        """Start events information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for start_event in process.findall('xmlns:startEvent', self.namespace):
                start_id = start_event.get('id')
                start_name = start_event.get('name')
                values.append(dict(start_id=start_id, start_name=start_name))
        return values

    def read_end_events(self):
        """End events information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for end_event in process.findall('xmlns:endEvent', self.namespace):
                end_id = end_event.get('id')
                end_name = end_event.get('name')
                values.append(dict(end_id=end_id, end_name=end_name))
        return values

    def read_intermediate_catch_events(self):
        """Intermediate catch events information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for timer_event in process.findall('xmlns:intermediateCatchEvent', self.namespace):
                timer_id = timer_event.get('id')
                timer_name = timer_event.get('name')
                values.append(dict(timer_id=timer_id, timer_name=timer_name))
        return values

    def read_sequence_flows(self):
        """Sequence flows information from the model."""
        values = []
        for process in self._root.findall('xmlns:process', self.namespace):
            for sequence in process.findall('xmlns:sequenceFlow', self.namespace):
                sf_id = sequence.get('id')
                source = sequence.get('sourceRef')
                target = sequence.get('targetRef')
                values.append(dict(sf_id=sf_id, source=source, target=target))
        return values

    def serialize_model(self) -> dict:
        ns = {'qbp': QBP_NAMESPACE_URI}

        model_xml = ET.tostring(self._root.find("qbp:processSimulationInfo", namespaces=ns))
        model_xml = model_xml.decode()
        model_xml = model_xml.replace(ns['qbp'], 'qbp')
        model_xml = bytes(model_xml, 'utf-8')
        model_xml = xtd.parse(model_xml)

        info = "qbp:processSimulationInfo"

        model_xml[info]['arrival_rate'] = (
            model_xml[info].pop('qbp:arrivalRateDistribution'))
        model_xml[info]['arrival_rate']['dname'] = (
            model_xml[info]['arrival_rate'].pop('@type'))
        model_xml[info]['arrival_rate']['dparams'] = dict()
        model_xml[info]['arrival_rate']['dparams']['arg1'] = (
            model_xml[info]['arrival_rate'].pop('@arg1'))
        model_xml[info]['arrival_rate']['dparams']['arg2'] = (
            model_xml[info]['arrival_rate'].pop('@arg2'))
        model_xml[info]['arrival_rate']['dparams']['mean'] = (
            model_xml[info]['arrival_rate'].pop('@mean'))
        model_xml[info]['arrival_rate'].pop('qbp:timeUnit')

        tags = {
            'element': 'elements_data',
            'resource': 'resource_pool',
            'sequenceFlow': 'sequences',
            'timetable': 'time_table'
        }

        for k, v in tags.items():
            element = model_xml[info]['qbp:' + k + 's']["qbp:" + k]
            model_xml[info].pop('qbp:' + k + 's')
            model_xml[info][v] = element
        model_xml[info]['instances'] = (
            model_xml[info].pop('@processInstances'))
        model_xml[info]['start_time'] = (
            model_xml[info].pop('@startDateTime'))
        model_xml[info].pop('@currency')
        model_xml[info].pop('@id')
        model_xml[info].pop('@xmlns:qbp')
        element = model_xml[info]
        model_xml.pop(info)
        model_xml = element

        activities = {
            x['task_id']: x['task_name']
            for x in self.read_activities()
        }

        for element in model_xml['elements_data']:
            element['elementid'] = element.pop('@elementId')
            element['id'] = element.pop('@id')
            element['arg1'] = element['qbp:durationDistribution']['@arg1']
            element['arg2'] = element['qbp:durationDistribution']['@arg2']
            element['mean'] = element['qbp:durationDistribution']['@mean']
            element['type'] = element['qbp:durationDistribution']['@type']
            element['resource'] = element['qbp:resourceIds']['qbp:resourceId']
            element['name'] = activities[element['elementid']]
            element.pop('qbp:durationDistribution')
            element.pop('qbp:resourceIds')

        sequence_flows = self.read_sequence_flows()

        for element in model_xml['sequences']:
            element['elementid'] = element.pop('@elementId')
            element['prob'] = element.pop('@executionProbability')
            seq = list(filter(lambda x: x['sf_id'] == element['elementid'], sequence_flows))[0]
            element['gatewayid'] = seq['source']
            element['out_path_id'] = seq['target']

        return model_xml

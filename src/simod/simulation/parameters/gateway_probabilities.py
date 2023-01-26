import xml.etree.ElementTree as ET
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import List

import pandas as pd

from simod.cli_formatter import print_notice, print_step
from simod.configuration import GatewayProbabilitiesDiscoveryMethod
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.prosimos_bpm_graph import BPMNGraph


@dataclass
class PathProbability:
    path_id: str
    probability: float

    def to_dict(self):
        """Dictionary compatible with Prosimos."""
        return {'path_id': self.path_id, 'value': self.probability}


@dataclass
class GatewayProbabilities:
    """Gateway branching probabilities for Prosimos."""
    gateway_id: str
    outgoing_paths: List[PathProbability]

    def to_dict(self):
        """Dictionary compatible with Prosimos."""
        return {'gateway_id': self.gateway_id, 'probabilities': [p.to_dict() for p in self.outgoing_paths]}


def mine_gateway_probabilities(
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        gateways_probability_type: GatewayProbabilitiesDiscoveryMethod) -> List[GatewayProbabilities]:
    bpmn_graph = BPMNGraph.from_bpmn_path(bpmn_path)

    # downstream functions work on list of traces instead of dataframe
    log_records = log.to_dict('records')
    cases = list(set([x[log_ids.case] for x in log_records]))
    traces = []
    for case in cases:
        order_key = log_ids.start_time
        trace = sorted(
            list(filter(lambda x, case_=case: (x[log_ids.case] == case_),
                        log_records)),
            key=itemgetter(order_key))
        traces.append(trace)

    sequences = _discover_with_gateway_management(traces, log_ids, bpmn_graph, gateways_probability_type)

    return _prosimos_gateways_probabilities(bpmn_path, sequences)


def _discover_with_gateway_management(
        log_traces: list,
        log_ids: EventLogIDs,
        bpmn_graph: BPMNGraph,
        gate_management: GatewayProbabilitiesDiscoveryMethod) -> list:
    if isinstance(gate_management, list) and len(gate_management) >= 1:
        print_notice(
            f'A list of gateway management options was provided: {gate_management}, taking the first option: {gate_management[0]}')
        gate_management = gate_management[0]

    print_step(f'Mining gateway probabilities with {gate_management}')
    if gate_management is GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE:
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_equiprobable()
    elif gate_management is GatewayProbabilitiesDiscoveryMethod.DISCOVERY:
        arcs_frequencies = _compute_sequence_flow_frequencies(log_traces, log_ids, bpmn_graph)
        gateways_branching = bpmn_graph.compute_branching_probability_alternative_discovery(arcs_frequencies)
    else:
        raise ValueError(
            f'Only GatewayManagement.DISCOVERY and .EQUIPROBABLE are supported, got {gate_management}, '
            f'{type(gate_management)}')

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return sequences


def _compute_sequence_flow_frequencies(log_traces: list, log_ids: EventLogIDs, bpmn_graph: BPMNGraph):
    flow_arcs_frequency = dict()
    for trace in log_traces:
        task_sequence = [event[log_ids.activity] for event in trace]
        bpmn_graph.replay_trace(task_sequence, flow_arcs_frequency)
    return flow_arcs_frequency


def _prosimos_gateways_probabilities(bpmn_path, sequences) -> List[GatewayProbabilities]:
    gateways_branching = {}
    reverse_map = {}

    tree = ET.parse(bpmn_path)
    root = tree.getroot()
    bpmn_element_ns = {'xmlns': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    for process in root.findall('xmlns:process', bpmn_element_ns):
        for xmlns_key in ['xmlns:exclusiveGateway', 'xmlns:inclusiveGateway']:
            for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                if bpmn_element.attrib["gatewayDirection"] == "Diverging":
                    gateway_id = bpmn_element.attrib["id"]
                    gateways_branching[gateway_id] = {}
                    for outgoing_flow in bpmn_element.findall("xmlns:outgoing", bpmn_element_ns):
                        outgoing_node_id = outgoing_flow.text.strip()
                        gateways_branching[gateway_id][outgoing_node_id] = 0
                        reverse_map[outgoing_node_id] = gateway_id

    for flow in sequences:
        flow_id = flow['elementid']
        probability = flow['prob']
        gateway_id = reverse_map[flow_id]
        gateways_branching[gateway_id][flow_id] = probability

    # converting to list for Prosimos

    result = []
    for gateway_id in gateways_branching:
        outgoing_paths = [
            PathProbability(outgoing_node, gateways_branching[gateway_id][outgoing_node])
            for outgoing_node in gateways_branching[gateway_id]]

        result.append(GatewayProbabilities(gateway_id, outgoing_paths))

    return result

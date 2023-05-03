from dataclasses import dataclass
from pathlib import Path
from typing import List
from xml.etree import ElementTree

import pandas as pd
from pix_utils.log_ids import EventLogIDs

from simod.cli_formatter import print_step
from simod.settings.control_flow_settings import GatewayProbabilitiesMethod
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
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        gateways_probability_type: GatewayProbabilitiesMethod) -> List[GatewayProbabilities]:
    print_step(f'Mining gateway probabilities with {gateways_probability_type}')
    # Read BPMN model
    bpmn_graph = BPMNGraph.from_bpmn_path(bpmn_path)
    # Discover gateway probabilities depending on the type
    if gateways_probability_type is GatewayProbabilitiesMethod.EQUIPROBABLE:
        gateways_branching = bpmn_graph.compute_equiprobable_gateway_probabilities()
    elif gateways_probability_type is GatewayProbabilitiesMethod.DISCOVERY:
        # Preprocess log: transform log to list of traces (list of activities)
        traces = []
        for _, events in event_log.groupby(log_ids.case):
            traces += [events.sort_values([log_ids.start_time, log_ids.end_time])[log_ids.activity].tolist()]
        # Discover the frequency of each gateway branch with replay
        arcs_frequencies = _compute_sequence_flow_frequencies(traces, bpmn_graph)
        gateways_branching = bpmn_graph.discover_gateway_probabilities(arcs_frequencies)
    else:
        # Error, wrong method
        raise ValueError(
            f'Only GatewayProbabilitiesMethod.DISCOVERY and GatewayProbabilitiesMethod.EQUIPROBABLE are supported, '
            f'got {gateways_probability_type} ({type(gateways_probability_type)}).'
        )

    sequences = []
    for gateway_id in gateways_branching:
        for seqflow_id in gateways_branching[gateway_id]:
            probability = gateways_branching[gateway_id][seqflow_id]
            sequences.append({'elementid': seqflow_id, 'prob': probability})

    return _prosimos_gateways_probabilities(bpmn_path, sequences)


def _compute_sequence_flow_frequencies(log_traces: list, bpmn_graph: BPMNGraph):
    flow_arcs_frequency = dict()
    for trace in log_traces:
        bpmn_graph.replay_trace(trace, flow_arcs_frequency)
    return flow_arcs_frequency


def _prosimos_gateways_probabilities(bpmn_path, sequences) -> List[GatewayProbabilities]:
    gateways_branching = {}
    reverse_map = {}

    tree = ElementTree.parse(bpmn_path)
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

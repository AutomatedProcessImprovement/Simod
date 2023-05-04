from dataclasses import dataclass
from pathlib import Path
from typing import List

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


def compute_gateway_probabilities(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        bpmn_path: Path,
        gateways_probability_type: GatewayProbabilitiesMethod
) -> List[GatewayProbabilities]:
    print_step(f'Mining gateway probabilities with {gateways_probability_type}')
    # Read BPMN model
    bpmn_graph = BPMNGraph.from_bpmn_path(bpmn_path)
    # Discover gateway probabilities depending on the type
    if gateways_probability_type is GatewayProbabilitiesMethod.EQUIPROBABLE:
        gateway_probabilities = bpmn_graph.compute_equiprobable_gateway_probabilities()
    elif gateways_probability_type is GatewayProbabilitiesMethod.DISCOVERY:
        # Discover the frequency of each gateway branch with replay
        arcs_frequencies = dict()
        for _, events in event_log.groupby(log_ids.case):
            # Transform to list of activity labels
            trace = events.sort_values([log_ids.start_time, log_ids.end_time])[log_ids.activity].tolist()
            # Replay updating arc frequencies
            bpmn_graph.replay_trace(trace, arcs_frequencies)
        # Obtain gateway path probabilities based on arc frequencies
        gateway_probabilities = bpmn_graph.discover_gateway_probabilities(arcs_frequencies)
    else:
        # Error, wrong method
        raise ValueError(
            f'Only GatewayProbabilitiesMethod.DISCOVERY and GatewayProbabilitiesMethod.EQUIPROBABLE are supported, '
            f'got {gateways_probability_type} ({type(gateways_probability_type)}).'
        )

    return _translate_to_prosimos_format(gateway_probabilities)


def _translate_to_prosimos_format(gateway_probabilities) -> List[GatewayProbabilities]:
    # Transform to prosimos list format
    prosimos_gateway_probabilities = [
        GatewayProbabilities(
            gateway_id,
            [
                PathProbability(outgoing_node, gateway_probabilities[gateway_id][outgoing_node])
                for outgoing_node in gateway_probabilities[gateway_id]
            ]
        )
        for gateway_id in gateway_probabilities
    ]
    # Return prosimos format
    return prosimos_gateway_probabilities

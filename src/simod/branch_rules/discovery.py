import pandas as pd
from typing import List

from simod.branch_rules.types import BranchRules

from pix_framework.io.event_log import EventLogIDs
from pix_framework.discovery.gateway_probabilities import GatewayProbabilities
from pix_framework.discovery.gateway_conditions.gateway_conditions import discover_gateway_conditions


def discover_branch_rules(bpmn_graph, log: pd.DataFrame, log_ids: EventLogIDs, f_score=0.7) -> list[BranchRules]:
    """
    Discover branch_rules from a log.
    """
    rules = discover_gateway_conditions(bpmn_graph, log, log_ids, f_score_threshold=f_score)

    rules = list(map(lambda x: BranchRules.from_dict(x), rules))

    return rules


def map_branch_rules_to_flows(gateway_probabilities: List[GatewayProbabilities], branch_rules: List[BranchRules]):
    condition_lookup = {rule.id: rule for rule in branch_rules}

    for gateway in gateway_probabilities:
        for path in gateway.outgoing_paths:
            if path.path_id in condition_lookup:
                path.condition_id = condition_lookup[path.path_id].id

    return gateway_probabilities

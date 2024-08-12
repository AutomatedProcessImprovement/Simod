import pandas as pd
from pix_framework.discovery.gateway_conditions.gateway_conditions import discover_gateway_conditions
from pix_framework.io.event_log import EventLogIDs

from simod.branch_rules.types import BranchRules


def discover_branch_rules(bpmn_graph, log: pd.DataFrame, log_ids: EventLogIDs) -> list[BranchRules]:
    """
    Discover batching _rules from a log.
    The enabled_time column is required. If it is missing, it will be estimated using the start-time-estimator.
    """
    rules = discover_gateway_conditions(bpmn_graph, log, log_ids)

    rules = list(map(lambda x: BranchRules.from_dict(x), rules))

    return rules


def map_branch_rules_to_flows(gateway_probabilities, branch_rules):
    conditions_lookup = {cond['id']: cond for cond in branch_rules}

    for gateway in gateway_probabilities:
        probabilities = gateway['probabilities']

        for prob in probabilities:
            flow_id = prob['path_id']

            if flow_id in conditions_lookup:
                prob['condition_id'] = flow_id

    return gateway_probabilities

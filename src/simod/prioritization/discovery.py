import pandas as pd
from pix_framework.log_ids import EventLogIDs
from prioritization_discovery.discovery import discover_priority_rules

from simod.event_log.utilities import add_enabled_time_if_missing
from .types import PrioritizationRule
from ..case_attributes.types import CaseAttribute


def discover_prioritization_rules(
    log: pd.DataFrame, log_ids: EventLogIDs, case_attributes: list[CaseAttribute]
) -> list[PrioritizationRule]:
    """
    Discover prioritization rules from a log.
    The enabled_time column is required. If it is missing, it will be estimated using the start-time-estimator.
    """
    log = add_enabled_time_if_missing(log, log_ids)

    case_attribute_names = list(map(lambda x: x.name, case_attributes))

    rules = discover_priority_rules(
        event_log=log,
        attributes=case_attribute_names,
    )

    rules = list(map(PrioritizationRule.from_prosimos, rules))

    return rules

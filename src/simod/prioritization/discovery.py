import pandas as pd
from pix_framework.discovery.prioritization.discovery import discover_priority_rules
from pix_framework.io.event_log import EventLogIDs

from ..data_attributes.types import CaseAttribute
from .types import PrioritizationRule


def discover_prioritization_rules(
    log: pd.DataFrame, log_ids: EventLogIDs, case_attributes: list[CaseAttribute]
) -> list[PrioritizationRule]:
    """
    Discover prioritization rules from a log.
    The enabled_time column is required. If it is missing, it will be estimated using the start-time-estimator.
    """
    case_attribute_names = list(map(lambda x: x.name, case_attributes))

    rules = discover_priority_rules(
        event_log=log.rename(  # Rename columns for hardcoded discovery package
            {log_ids.enabled_time: "enabled_time", log_ids.start_time: "start_time", log_ids.resource: "Resource"},
            axis=1,
        ),
        attributes=case_attribute_names,
    )

    rules = list(map(PrioritizationRule.from_prosimos, rules))

    return rules

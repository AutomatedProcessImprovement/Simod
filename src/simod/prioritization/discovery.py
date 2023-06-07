import pandas as pd
from case_attribute_discovery.discovery import discover_case_attributes
from pix_framework.log_ids import EventLogIDs
from prioritization_discovery.discovery import discover_priority_rules

from simod.event_log.utilities import add_enabled_time_if_missing
from .types import PrioritizationLevel


def discover_prioritization_rules(log: pd.DataFrame, log_ids: EventLogIDs) -> list[PrioritizationLevel]:
    """
    Discover prioritization rules from a log.
    The enabled_time column is required. If it is missing, it will be estimated using the start-time-estimator.
    """
    log = add_enabled_time_if_missing(log, log_ids)

    case_attributes = get_case_attributes(log, log_ids)
    case_attribute_names = list(map(lambda x: x["name"], case_attributes))

    rules = discover_priority_rules(
        event_log=log,
        attributes=case_attribute_names,
    )

    rules = list(map(PrioritizationLevel.from_dict, rules))

    return rules


def get_case_attributes(log: pd.DataFrame, log_ids: EventLogIDs) -> list[dict]:
    """
    Discover case attributes from a log ignoring common non-case columns.
    """
    return discover_case_attributes(
        event_log=log,
        log_ids=log_ids,
        avoid_columns=[
            log_ids.case,
            log_ids.activity,
            log_ids.enabled_time,
            log_ids.start_time,
            log_ids.end_time,
            log_ids.resource,
        ],
    )

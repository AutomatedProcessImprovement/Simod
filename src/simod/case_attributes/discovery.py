import pandas as pd
from case_attribute_discovery.discovery import discover_case_attributes as discover_case_attributes_
from pix_framework.log_ids import EventLogIDs

from simod.case_attributes.types import CaseAttribute


def discover_case_attributes(log: pd.DataFrame, log_ids: EventLogIDs) -> list[CaseAttribute]:
    """
    Discover case attributes from a log ignoring common non-case columns.
    """
    attributes = discover_case_attributes_(
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
        confidence_threshold=0.95,
    )

    attributes = list(map(CaseAttribute.from_dict, attributes))

    return attributes

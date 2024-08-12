import pandas as pd
from pix_framework.discovery.attributes.attribute_discovery import discover_attributes
from pix_framework.io.event_log import EventLogIDs

from simod.data_attributes.types import GlobalAttribute, CaseAttribute, EventAttribute


def discover_data_attributes(log: pd.DataFrame, log_ids: EventLogIDs) -> (list[CaseAttribute], list[GlobalAttribute], list[EventAttribute]):
    """
    Discover data attributes from a log ignoring common non-case columns.
    """
    attributes = discover_attributes(
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

    global_attributes = list(map(GlobalAttribute.from_dict, attributes["global_attributes"]))
    case_attributes = list(map(CaseAttribute.from_dict, attributes["case_attributes"]))
    event_attributes = list(map(EventAttribute.from_dict, attributes["event_attributes"]))

    return global_attributes, case_attributes, event_attributes


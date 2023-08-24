import pandas as pd
from pix_framework.discovery.batch_processing.batch_characteristics import discover_batch_processing_and_characteristics
from pix_framework.io.event_log import EventLogIDs

from simod.batching.types import BatchingRule


def discover_batching_rules(log: pd.DataFrame, log_ids: EventLogIDs) -> list[BatchingRule]:
    """
    Discover batching _rules from a log.
    The enabled_time column is required. If it is missing, it will be estimated using the start-time-estimator.
    """
    rules = discover_batch_processing_and_characteristics(
        event_log=log, log_ids=log_ids, batch_min_size=3, max_sequential_gap=pd.Timedelta("10m")
    )

    rules = list(map(lambda x: BatchingRule.from_dict(x), rules))

    return rules

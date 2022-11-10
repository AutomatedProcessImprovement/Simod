import pandas as pd

from log_similarity_metrics.absolute_timestamps import absolute_timestamps_emd, discretize_to_hour
from log_similarity_metrics.config import AbsoluteTimestampType
from simod.event_log.column_mapping import EventLogIDs


def get_absolute_timestamps_emd(
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
    emd = absolute_timestamps_emd(
        event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, AbsoluteTimestampType.BOTH, discretize_to_hour)
    return emd

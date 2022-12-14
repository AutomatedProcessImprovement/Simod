import pandas as pd

from simod.discovery.distribution import get_best_distribution
from simod.event_log.column_mapping import EventLogIDs


def discover(log: pd.DataFrame, log_ids: EventLogIDs) -> dict:
    """
    Discovers case inter-arrival duration distribution for the event log.

    :param log: Event log.
    :param log_ids: Event log column IDs.
    :return: Distribution parameters.
    """
    inter_arrival_durations = []

    start_times = []
    for (case_id, group) in log.groupby(log_ids.case):
        start_times.append(group[log_ids.start_time].min())

    start_times = sorted(start_times)

    for i in range(1, len(start_times)):
        delta = (start_times[i] - start_times[i - 1]).total_seconds()
        inter_arrival_durations.append(delta)

    return get_best_distribution(inter_arrival_durations)

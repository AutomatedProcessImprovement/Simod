import datetime

import pandas as pd
from log_distance_measures.absolute_event_distribution import (
    absolute_event_distribution_distance,
    discretize_to_hour,
)
from log_distance_measures.circadian_event_distribution import (
    circadian_event_distribution_distance,
)
from log_distance_measures.config import AbsoluteTimestampType
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import (
    cycle_time_distribution_distance,
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from pix_utils.log_ids import EventLogIDs
from simod.settings.common_settings import Metric


def compute_metric(
    metric: Metric,
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    """Computes an event log metric for two event logs, e.g., similarity, distance, etc.

    :param metric: The metric to compute.
    :param event_log_1: First event log.
    :param event_log_1_ids: Column names of the first event log.
    :param event_log_2: Second event log.
    :param event_log_2_ids: Column names of the second event log.
    :return: The computed metric.
    """

    if metric is Metric.DL:
        result = get_dl(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    elif metric is Metric.N_GRAM_DISTANCE:
        result = get_n_grams_distribution_distance(
            event_log_1, event_log_1_ids, event_log_2, event_log_2_ids
        )
    elif metric is Metric.CIRCADIAN_EMD:
        result = get_circadian_emd(
            event_log_1, event_log_1_ids, event_log_2, event_log_2_ids
        )
    elif metric is Metric.ABSOLUTE_HOURLY_EMD:
        result = get_absolute_hourly_emd(
            event_log_1, event_log_1_ids, event_log_2, event_log_2_ids
        )
    elif metric is Metric.CYCLE_TIME_EMD:
        result = get_cycle_time_emd(
            event_log_1, event_log_1_ids, event_log_2, event_log_2_ids
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return result


def get_absolute_hourly_emd(
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    """Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing
    the timestamps by absolute hour.

    :param event_log_1: First event log.
    :param event_log_1_ids: Column names of the first event log.
    :param event_log_2: Second event log.
    :param event_log_2_ids: Column names of the second event log.
    :return: The absolute timestamps EMD.
    """

    emd = absolute_event_distribution_distance(
        event_log_1,
        event_log_1_ids,
        event_log_2,
        event_log_2_ids,
        AbsoluteTimestampType.BOTH,
        discretize_to_hour,
    )
    return emd


def get_cycle_time_emd(
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    """Distance measure computing how different the cycle time discretized histograms of two event logs are.

    :param event_log_1: First event log.
    :param event_log_1_ids: Column names of the first event log.
    :param event_log_2: Second event log.
    :param event_log_2_ids: Column names of the second event log.
    :return: The cycle time EMD.
    """
    emd = cycle_time_distribution_distance(
        event_log_1,
        event_log_1_ids,
        event_log_2,
        event_log_2_ids,
        datetime.timedelta(hours=1),
    )
    return emd


def get_circadian_emd(
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the timestamps of two event logs are, comparing all
    the instants recorded in the same weekday together, and discretizing them to the hour in the day.
    :param event_log_1:
    :param event_log_1_ids:
    :param event_log_2:
    :param event_log_2_ids:
    :return: The circadian EMD.
    """
    emd = circadian_event_distribution_distance(
        event_log_1,
        event_log_1_ids,
        event_log_2,
        event_log_2_ids,
        AbsoluteTimestampType.BOTH,
    )
    return emd


def get_n_grams_distribution_distance(
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    """
    Distance measure between two event logs computing the difference in the frequencies of the n-grams observed in
    the event logs (being the n-grams of an event log all the groups of n consecutive elements observed in it).

    :param event_log_1: first event log.
    :param event_log_1_ids: IDs of the first event log.
    :param event_log_2: second event log.
    :param event_log_2_ids: IDs of the second event log.
    :return: The MAE between the frequency of trigrams occurring in one log vs the other.
    """
    mae = n_gram_distribution_distance(
        event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, 3
    )
    return mae


def get_dl(
    event_log_1: pd.DataFrame,
    event_log_1_ids: EventLogIDs,
    event_log_2: pd.DataFrame,
    event_log_2_ids: EventLogIDs,
) -> float:
    cfld = control_flow_log_distance(
        event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, True
    )
    return cfld

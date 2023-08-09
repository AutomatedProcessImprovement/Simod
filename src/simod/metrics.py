import datetime

import pandas as pd
from log_distance_measures.absolute_event_distribution import (
    absolute_event_distribution_distance,
    discretize_to_hour,
)
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import (
    circadian_event_distribution_distance,
)
from log_distance_measures.config import AbsoluteTimestampType
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import (
    cycle_time_distribution_distance,
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from pix_framework.io.event_log import EventLogIDs

from simod.settings.common_settings import Metric


def compute_metric(
    metric: Metric,
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """Computes the distance between an original (test) event log and a simulated one.

    :param metric: The metric to compute.
    :param original_log: Original event log.
    :param original_log_ids: Column names of the original event log.
    :param simulated_log: Simulated event log.
    :param simulated_log_ids: Column names of the simulated event log.

    :return: The computed metric.
    """

    if metric is Metric.DL:
        result = get_dl(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.TWO_GRAM_DISTANCE:
        result = get_n_grams_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, 2)
    elif metric is Metric.THREE_GRAM_DISTANCE:
        result = get_n_grams_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, 3)
    elif metric is Metric.CIRCADIAN_EMD:
        result = get_circadian_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.ARRIVAL_EMD:
        result = get_arrival_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.RELATIVE_EMD:
        result = get_relative_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.ABSOLUTE_EMD:
        result = get_absolute_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.CYCLE_TIME_EMD:
        result = get_cycle_time_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return result


def get_absolute_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing
    the timestamps by absolute hour.
    """

    emd = absolute_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
        discretize_to_hour,
    )
    return emd


def get_cycle_time_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the cycle time discretized histograms of two event logs are.
    """
    emd = cycle_time_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        datetime.timedelta(hours=1),
    )
    return emd


def get_circadian_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the timestamps of two event logs are, comparing all
    the instants recorded in the same weekday together (e.g., Monday), and discretizing them to the hour in the day.
    """
    emd = circadian_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
    )
    return emd


def get_arrival_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the case arrivals of two event logs are.
    """
    emd = case_arrival_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
    )
    return emd


def get_relative_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the distribution of the events with each case (i.e., relative to their
    start) of two event logs are.
    """
    emd = relative_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
    )
    return emd


def get_n_grams_distribution_distance(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
    n: int = 3,
) -> float:
    """
    Distance measure between two event logs computing the difference in the frequencies of the n-grams observed in
    the event logs (being the n-grams of an event log all the groups of n consecutive elements observed in it).
    :return: The MAE between the frequency of trigrams occurring in one log vs the other.
    """
    mae = n_gram_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, n=n)
    return mae


def get_dl(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    cfld = control_flow_log_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, True)
    return cfld

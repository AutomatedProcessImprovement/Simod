import datetime

import pandas as pd

from log_similarity_metrics.absolute_event_distribution import absolute_event_distribution_distance, discretize_to_hour
from log_similarity_metrics.circadian_event_distribution import circadian_event_distribution_distance
from log_similarity_metrics.config import AbsoluteTimestampType
from log_similarity_metrics.cycle_time_distribution import cycle_time_distribution_distance
from simod.configuration import Metric
from simod.event_log.column_mapping import EventLogIDs
from simod.metrics.tsd_evaluator import TimedStringDistanceEvaluator


def compute_metric(
        metric: Metric,
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
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
    elif metric is Metric.CIRCADIAN_EMD:
        result = get_circadian_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    elif metric is Metric.ABSOLUTE_HOURLY_EMD:
        result = get_absolute_hourly_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    elif metric is Metric.CYCLE_TIME_EMD:
        result = get_cycle_time_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    else:
        raise ValueError(f'Unsupported metric: {metric}')

    return result


def get_absolute_hourly_emd(
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
    """Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing
    the timestamps by absolute hour.

    :param event_log_1: First event log.
    :param event_log_1_ids: Column names of the first event log.
    :param event_log_2: Second event log.
    :param event_log_2_ids: Column names of the second event log.
    :return: The absolute timestamps EMD.
    """

    emd = absolute_event_distribution_distance(
        event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, AbsoluteTimestampType.BOTH, discretize_to_hour)
    return emd


def get_cycle_time_emd(
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
    """Distance measure computing how different the cycle time discretized histograms of two event logs are.

    :param event_log_1: First event log.
    :param event_log_1_ids: Column names of the first event log.
    :param event_log_2: Second event log.
    :param event_log_2_ids: Column names of the second event log.
    :return: The cycle time EMD.
    """
    emd = cycle_time_distribution_distance(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids,
                                           datetime.timedelta(hours=1))
    return emd


def get_circadian_emd(
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
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
        event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, AbsoluteTimestampType.BOTH)
    return emd


def get_dl(
        event_log_1: pd.DataFrame,
        event_log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        event_log_2_ids: EventLogIDs) -> float:
    evaluator = TimedStringDistanceEvaluator(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    return evaluator.measure_distance()

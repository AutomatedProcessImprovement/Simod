import datetime

import pandas as pd

from log_similarity_metrics.absolute_timestamps import absolute_timestamps_emd, discretize_to_hour
from log_similarity_metrics.config import AbsoluteTimestampType
from log_similarity_metrics.cycle_times import cycle_time_emd
from simod.configuration import Metric
from simod.event_log.column_mapping import EventLogIDs


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
    from simod.analyzers.sim_evaluator import SimilarityEvaluator

    if metric in [
        Metric.TSD,
        Metric.DL,
        Metric.MAE,
        Metric.DL_MAE,
        Metric.LOG_MAE,
        Metric.HOUR_EMD,
        Metric.DAY_EMD,
        Metric.DAY_HOUR_EMD,
        Metric.CAL_EMD
    ]:
        evaluator = SimilarityEvaluator(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
        result = evaluator.measure_distance(metric)['similarity']
    elif metric is Metric.ABSOLUTE_HOURLY_EMD:
        result = get_absolute_timestamps_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    elif metric is Metric.CYCLE_TIME_EMD:
        result = get_cycle_time_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids)
    else:
        raise ValueError(f'Unsupported metric: {metric}')

    return result


def get_absolute_timestamps_emd(
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

    emd = absolute_timestamps_emd(
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
    emd = cycle_time_emd(event_log_1, event_log_1_ids, event_log_2, event_log_2_ids, datetime.timedelta(hours=1))
    return emd

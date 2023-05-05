from typing import List

import pandas as pd
from bpdfr_simulation_engine.resource_calendar import CalendarFactory
from pix_utils.log_ids import EventLogIDs
from pix_utils.statistics.distribution import get_best_fitting_distribution, get_observations_histogram

from simod.simulation.parameters.calendar import Calendar, Timetable
from simod.utilities import nearest_divisor_for_granularity


def discover_case_arrival_calendar(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        granularity=60
) -> Calendar:
    """
    Discover weekly calendar for the arrival of new cases, i.e., the periods of times in each day when
    new cases arrive to the system.

    :param event_log: event log to model the case arrivals from.
    :param log_ids: Event log column IDs.
    :param granularity: number of minutes to take as minimum available interval surrounding each
                        observed arrival.

    :return: weekly calendar of case arrivals.
    """
    # Correct granularity if not divisor of 1440 (minutes in a day)
    if 1440 % granularity != 0:
        granularity = nearest_divisor_for_granularity(granularity)
    # Create calendar discoverer and store arrivals
    calendar_factory = CalendarFactory(granularity)
    for case_id, events in event_log.groupby(by=log_ids.case):
        resource = "system"  # Assign all arrivals to the same resource
        activity = "case_arrival"  # Assign same activity label to all arrivals
        case_arrival = events[log_ids.start_time].min()
        calendar_factory.check_date_time(resource, activity, case_arrival)
    # Discover calendar for the case arrivals
    calendars = calendar_factory.build_weekly_calendars(min_confidence=0.1, desired_support=0.7, min_participation=0.4)
    calendar = Calendar(
        id='Case arrival calendar',
        name='Case arrival calendar',
        timetables=Timetable.from_list_of_dicts(calendars["system"].to_json())
    )
    # Return case arrival calendar
    return calendar


def discover_inter_arrival_distribution(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        filter_outliers: bool = True
) -> dict:
    """
    Discover case inter-arrival duration distribution for the event log.

    :param event_log: Event log.
    :param log_ids: Event log column IDs.
    :param filter_outliers: flag to remove outlier inter-arrival times.
    :return: Duration distribution for the inter-arrival times.
    """
    # Get the durations between each two consecutive arrivals
    inter_arrival_durations = _get_inter_arrival_times(event_log, log_ids)
    # Get the best distribution fitting the inter-arrival durations
    arrival_distribution = get_best_fitting_distribution(
        data=inter_arrival_durations,
        filter_outliers=filter_outliers
    )
    # Return it
    return arrival_distribution.to_prosimos_distribution()


def get_observed_inter_arrival_distribution(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        num_bins: int = 20,
        filter_outliers: bool = True
) -> dict:
    """
    Get the distribution of observed inter-arrival times (CDF and bin midpoints of their histogram).

    :param event_log: event log to extract the arrivals.
    :param log_ids: column mapping IDs for the event log.
    :param num_bins: number of bins of the build histogram.
    :param filter_outliers: flag to remove outlier inter-arrival times.
    :return: CDF and bin midpoints of the histogram modelling the inter-arrivals.
    """
    # Get the durations between each two consecutive arrivals
    inter_arrival_durations = _get_inter_arrival_times(event_log, log_ids)
    # Compute the CDF and BINs of the observations histogram
    arrival_distribution = get_observations_histogram(
        data=inter_arrival_durations,
        num_bins=num_bins,
        filter_outliers=filter_outliers
    )
    # Return custom histogram distribution
    return arrival_distribution


def _get_inter_arrival_times(event_log: pd.DataFrame, log_ids: EventLogIDs) -> List[float]:
    # Get the arrival times from the event log
    arrival_times = []
    for case_id, events in event_log.groupby(by=log_ids.case):
        arrival_times += [events[log_ids.start_time].min()]
    # Sort them
    arrival_times.sort()
    # Compute durations between one arrival and the next one (inter-arrival durations)
    inter_arrival_durations = []
    last_arrival = None
    for arrival in arrival_times:
        if last_arrival:
            inter_arrival_durations += [(arrival - last_arrival).total_seconds()]
        last_arrival = arrival
    # Return list of inter-arrivals
    return inter_arrival_durations

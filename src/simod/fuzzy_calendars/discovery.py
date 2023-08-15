from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pix_framework.calendar.resource_calendar import int_week_days, str_week_days
from pix_framework.discovery.resource_activity_performances import ActivityResourceDistribution
from pix_framework.io.event_log import EventLogIDs
from pix_framework.statistics.distribution import DurationDistribution

from .factory import FuzzyFactory
from .proccess import Method, Process

# Type aliases for readability
ActivityID = str
ActivityName = str
ResourceName = str
ActivityResourceDistributionOrlenys = dict[ResourceName, dict[ActivityName, DurationDistribution]]
ActivityResources = dict[ActivityID, set[ResourceName]]
ActivityNameToIDMapping = dict[ActivityName, ActivityID]


class FuzzyInterval:
    _from_day: int
    _to_day: int
    _start_time: pd.Timestamp
    _end_time: pd.Timestamp
    probability: float

    def __init__(
        self,
        from_day: int,
        to_day: int,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        probability: float
    ):
        self._from_day = from_day
        self._to_day = to_day
        self._start_time = start_time
        self._end_time = end_time
        self.probability = probability

    @property
    def from_day(self):
        return int_week_days[self._from_day]

    @property
    def to_day(self):
        return int_week_days[self._to_day]

    @property
    def start_time(self):
        return self._start_time.strftime("%H:%M:%S")

    @property
    def end_time(self):
        return self._end_time.strftime("%H:%M:%S")

    def to_prosimos(self) -> dict:
        return {
            "from": self.from_day,
            "to": self.to_day,
            "beginTime": self.start_time,
            "endTime": self.end_time,
            "probability": self.probability,
        }

    @staticmethod
    def from_prosimos(interval: dict) -> "FuzzyInterval":
        return FuzzyInterval(
            from_day=str_week_days[interval["from"]],
            to_day=str_week_days[interval["to"]],
            start_time=pd.Timestamp(interval["beginTime"]),
            end_time=pd.Timestamp(interval["endTime"]),
            probability=interval["probability"],
        )


@dataclass
class FuzzyResourceCalendar:
    resource_id: str
    resource_name: str
    intervals: list[FuzzyInterval]
    workloads: list[FuzzyInterval]

    def to_prosimos(self):
        return {
            "id": self.resource_id,
            "name": self.resource_id,
            "time_periods": self.intervals,
            "workload_ratio": self.workloads,
        }

    @staticmethod
    def from_prosimos(calendar: dict) -> "FuzzyResourceCalendar":
        return FuzzyResourceCalendar(
            resource_id=calendar["id"],
            resource_name=calendar["name"],
            intervals=[FuzzyInterval.from_prosimos(i) for i in calendar["time_periods"]],
            workloads=[FuzzyInterval.from_prosimos(i) for i in calendar["workload_ratio"]],
        )


def _get_activities_resources(
    log: pd.DataFrame,
    log_ids: EventLogIDs,
    activities_ids_by_name: Optional[ActivityNameToIDMapping] = None,
) -> ActivityResources:
    activities_resources = {activity_name: set() for activity_name in log[log_ids.activity].unique()}
    for activity_name in activities_resources:
        activities_resources[activity_name] = set(
            log[log[log_ids.activity] == activity_name][log_ids.resource].unique()
        )

    # Use the provided mapping to convert activity names to activity IDs if it is provided
    if activities_ids_by_name is not None:
        activities_resources = {
            activities_ids_by_name[activity_name]: resources
            for activity_name, resources in activities_resources.items()
        }

    return activities_resources


def discovery_fuzzy_simulation_parameters(
    log: pd.DataFrame,
    log_ids: EventLogIDs,
    task_ids_by_name: ActivityNameToIDMapping,
    granularity=15,
    angle=0.0,
) -> tuple[list[FuzzyResourceCalendar], list[ActivityResourceDistribution]]:
    """
    Discovers fuzzy simulation parameters from an event log.
    NOTE: Enabled time must be present in the event log.
    """
    activity_resources = _get_activities_resources(log, log_ids)

    process = Process(granularity, activity_resources, log, log_ids, True, Method.TRAPEZOIDAL, angle=angle)
    fuzzy_factory = FuzzyFactory(process)

    # discovery
    process.fuzzy_calendars = fuzzy_factory.compute_resource_availability_calendars()
    activity_resource_distributions_orlenys = fuzzy_factory.compute_processing_times(process.fuzzy_calendars)

    # transform
    resource_calendars = _join_fuzzy_calendar_intervals(process.fuzzy_calendars, process.i_size)
    activity_resource_distributions_prosimos = _convert_fuzzy_activity_resource_distributions_to_prosimos(
        activity_resource_distributions_orlenys, activity_resources, task_ids_by_name
    )

    # convert to readable types
    resource_calendars_typed = [FuzzyResourceCalendar.from_prosimos(c) for c in resource_calendars]
    activity_resource_distributions = [
        ActivityResourceDistribution.from_dict(d) for d in activity_resource_distributions_prosimos
    ]

    return resource_calendars_typed, activity_resource_distributions


def _convert_fuzzy_activity_resource_distributions_to_prosimos(
    activity_resource_distributions: ActivityResourceDistributionOrlenys,
    activity_resources: ActivityResources,
    activities_names_to_ids: ActivityNameToIDMapping,
):
    distributions = []

    for activity_name in activity_resources:
        resources = []
        for resource_name in activity_resources[activity_name]:
            if resource_name not in activity_resource_distributions:
                continue

            distribution: DurationDistribution = activity_resource_distributions[resource_name][activity_name]
            distribution_prosimos = distribution.to_prosimos_distribution()

            resources.append(
                {
                    "resource_id": resource_name,
                    "distribution_name": distribution_prosimos["distribution_name"],
                    "distribution_params": distribution_prosimos["distribution_params"],
                }
            )

        distributions.append({"task_id": activities_names_to_ids[activity_name], "resources": resources})

    return distributions


def _join_fuzzy_calendar_intervals(fuzzy_calendars, i_size):
    resource_calendars = []
    for r_id in fuzzy_calendars:
        resource_calendars.append(
            {
                "id": "%s_timetable" % r_id,
                "name": "%s_timetable" % r_id,
                "time_periods": _sweep_line_intervals(fuzzy_calendars[r_id].res_absolute_prob, i_size),
                "workload_ratio": _sweep_line_intervals(fuzzy_calendars[r_id].res_relative_prob, i_size),
            }
        )
    return resource_calendars


def _sweep_line_intervals(prob_map, i_size):
    time_periods = []
    for w_day in int_week_days:
        joint_intervals = []
        c_prob = prob_map[w_day][0]
        first_i = 0
        for i in range(1, len(prob_map[w_day])):
            if c_prob != prob_map[w_day][i]:
                if c_prob != 0:
                    joint_intervals.append((first_i, i))
                first_i = i
                c_prob = prob_map[w_day][i]
        if c_prob != 0:
            joint_intervals.append((first_i, 0))
        for from_i, to_i in joint_intervals:
            time_periods.append(
                {
                    "from": int_week_days[w_day],
                    "to": int_week_days[w_day],
                    "beginTime": str(_interval_index_to_time(from_i, i_size, True).time()),
                    "endTime": str(_interval_index_to_time(to_i, i_size, True).time()),
                    "probability": prob_map[w_day][from_i],
                }
            )
    return time_periods


def _interval_index_to_time(i_index, i_size, is_start):
    from_time = datetime.strptime("00:00:00", "%H:%M:%S") + timedelta(minutes=(i_index * i_size))
    return from_time if is_start else from_time + timedelta(minutes=i_size)

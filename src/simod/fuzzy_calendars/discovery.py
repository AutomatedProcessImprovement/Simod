import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from bpdfr_discovery.log_parser import (
    discover_arrival_calendar,
    discover_arrival_time_distribution,
)
from pix_framework.discovery.resource_activity_performances import ActivityResourceDistribution
from pix_framework.statistics.distribution import DurationDistribution
from pix_framework.io.event_log import EventLogIDs
from prosimos.execution_info import TaskEvent, Trace
from prosimos.simulation_properties_parser import parse_simulation_model

from simod.fuzzy_calendars.factory import FuzzyFactory
from simod.fuzzy_calendars.proccess_info import Method, ProcInfo


class FuzzyTimeInterval:
    _start_time: pd.Timestamp
    _end_time: pd.Timestamp
    probability: float

    def __init__(self, start_time: pd.Timestamp, end_time: pd.Timestamp, probability: float):
        self._start_time = start_time
        self._end_time = end_time
        self.probability = probability

    @property
    def start_time(self):
        return self._start_time.strftime("%H:%M:%S")

    @property
    def end_time(self):
        return self._end_time.strftime("%H:%M:%S")

    def to_prosimos(self) -> dict:
        return {"begin_time": self.start_time, "end_time": self.end_time, "probability": self.probability}

    @staticmethod
    def from_prosimos(interval: dict) -> "FuzzyTimeInterval":
        return FuzzyTimeInterval(
            start_time=pd.Timestamp(interval["begin_time"]),
            end_time=pd.Timestamp(interval["end_time"]),
            probability=interval["probability"],
        )


@dataclass
class FuzzyDay:
    week_day: str
    in_day_intervals: list[FuzzyTimeInterval]

    def to_prosimos(self) -> dict:
        return {"week_day": self.week_day, "fuzzy_intervals": [i.to_prosimos() for i in self.in_day_intervals]}

    @staticmethod
    def from_prosimos(day: dict) -> "FuzzyDay":
        return FuzzyDay(
            week_day=day["week_day"],
            in_day_intervals=[FuzzyTimeInterval.from_prosimos(i) for i in day["fuzzy_intervals"]],
        )


@dataclass
class FuzzyResourceCalendar:
    resource_id: str
    intervals: list[FuzzyDay]
    workloads: list[FuzzyDay]

    def to_prosimos(self):
        return {
            "id": self.resource_id,
            "availability_probabilities": self.intervals,
            "workload_ratio": self.workloads,
        }

    @staticmethod
    def from_prosimos(calendar: dict) -> "FuzzyResourceCalendar":
        return FuzzyResourceCalendar(
            resource_id=calendar["id"],
            intervals=[FuzzyDay.from_prosimos(i) for i in calendar["availability_probabilities"]],
            workloads=[FuzzyDay.from_prosimos(i) for i in calendar["workload_ratio"]],
        )


def discovery_fuzzy_simulation_parameters(
    log: pd.DataFrame,
    log_ids: EventLogIDs,
    bpmn_path: Path,
    granularity=15,
    angle=0.0,
    min_prob=0.1,
) -> tuple[list[FuzzyResourceCalendar], list[ActivityResourceDistribution]]:
    traces = event_list_from_df(log, log_ids)
    bpmn_graph = parse_simulation_model(bpmn_path)

    p_info = ProcInfo(traces, bpmn_graph, granularity, True, Method.TRAPEZOIDAL, angle=angle)
    f_factory = FuzzyFactory(p_info)

    # discovery
    p_info.fuzzy_calendars = f_factory.compute_resource_availability_calendars(min_impact=min_prob)
    res_task_distr = f_factory.compute_processing_times(p_info.fuzzy_calendars)

    # transform
    resource_calendars = _join_fuzzy_calendar_intervals(p_info.fuzzy_calendars, p_info.i_size)
    activity_resource_distributions = _processing_times_json(res_task_distr, p_info.task_resources, p_info.bpmn_graph)

    # convert to readable types
    resource_calendars_typed = [FuzzyResourceCalendar.from_prosimos(c) for c in resource_calendars]
    activity_resource_distributions_typed = [
        ActivityResourceDistribution.from_dict(d) for d in activity_resource_distributions
    ]

    return resource_calendars_typed, activity_resource_distributions_typed


def build_fuzzy_calendars(
    csv_log_path: Path, bpmn_path: Path, json_path: Optional[Path] = None, i_size_minutes=15, angle=0.0, min_prob=0.1
):
    traces = event_list_from_csv(csv_log_path)
    bpmn_graph = parse_simulation_model(bpmn_path)

    p_info = ProcInfo(traces, bpmn_graph, i_size_minutes, True, Method.TRAPEZOIDAL, angle=angle)
    f_factory = FuzzyFactory(p_info)

    # 1) Discovering Resource Availability (Fuzzy Calendars)
    p_info.fuzzy_calendars = f_factory.compute_resource_availability_calendars(min_impact=min_prob)

    # 2) Discovering Resource Performance (resource-task distributions adjusted from the fuzzy calendars)
    res_task_distr = f_factory.compute_processing_times(p_info.fuzzy_calendars)

    # 3) Discovering Arrival Time Calendar -- Nothing New, just re-using the original Prosimos approach
    arrival_calend = discover_arrival_calendar(p_info.initial_events, 15, 0.1, 1.0)

    # 4) Discovering Arrival Time Distribution -- Nothing New, just re-using the original Prosimos approach
    arrival_dist = discover_arrival_time_distribution(p_info.initial_events, arrival_calend)

    # 5) Discovering Gateways Branching Probabilities -- Nothing New, just re-using the original Prosimos approach
    gateways_branching = bpmn_graph.compute_branching_probability(p_info.flow_arcs_frequency)

    simulation_params = {
        "resource_profiles": _build_resource_profiles(p_info),
        "arrival_time_distribution": _distribution_to_json(arrival_dist),
        "arrival_time_calendar": arrival_calend.to_json(),
        "gateway_branching_probabilities": _gateway_branching_to_json(gateways_branching),
        "task_resource_distribution": _processing_times_json(res_task_distr, p_info.task_resources, p_info.bpmn_graph),
        "resource_calendars": _join_fuzzy_calendar_intervals(p_info.fuzzy_calendars, p_info.i_size),
        "granule_size": {"value": i_size_minutes, "time_unit": "MINUTES"},
    }

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(simulation_params, f)

    return simulation_params


def event_list_from_csv(log_path) -> list[Trace]:
    def find_index(csv_row):
        i_map = dict()
        for i in range(0, len(csv_row)):
            i_map[csv_row[i]] = i
        return i_map

    try:
        with open(log_path, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            trace_list = list()
            trace_map = dict()
            i_map = dict()
            row_count = 0
            for row in csv_reader:
                if row_count > 0:
                    case_id = row[i_map["case_id"]]
                    event_info = TaskEvent(case_id, row[i_map["activity"]], row[i_map["resource"]])
                    if "enable_time" in i_map and len(row[i_map["enable_time"]]) > 0:
                        event_info.enabled_at = pd.to_datetime(row[i_map["enable_time"]])
                    event_info.started_at = pd.to_datetime(row[i_map["start_time"]])
                    event_info.completed_at = pd.to_datetime(row[i_map["end_time"]])
                    if case_id not in trace_map:
                        trace_map[case_id] = len(trace_list)
                        trace_list.append(Trace(case_id))
                    trace_list[trace_map[case_id]].event_list.append(event_info)
                else:
                    i_map = find_index(row)
                row_count += 1
            return trace_list
    except IOError:
        return list()


def event_list_from_df(log: pd.DataFrame, log_ids: EventLogIDs) -> list[Trace]:
    """
    Creates a list of Prosimos traces from an event log.
    """

    traces = {}
    cases = log[log_ids.case].unique()
    for case in cases:
        traces[case] = Trace(case)

    def compose_event_from_row(row) -> TaskEvent:
        task_event = TaskEvent(
            p_case=getattr(row, log_ids.case),
            task_id=getattr(row, log_ids.activity),
            resource_id=getattr(row, log_ids.resource),
        )
        task_event.started_at = getattr(row, log_ids.start_time)
        task_event.completed_at = getattr(row, log_ids.end_time)
        if log_ids.enabled_time in log.columns and len(getattr(row, log_ids.enabled_time)) > 0:
            task_event.enabled_at = getattr(row, log_ids.enabled_time)
        return task_event

    for case in cases:
        events = log[log[log_ids.case] == case]
        task_events = [compose_event_from_row(row) for row in events.itertuples()]
        trace = traces[case]
        trace.event_list = task_events

    trace_list = [trace for trace in traces.values() if trace is not None]

    return trace_list


def _processing_times_json(res_task_distr, task_resources, bpmn_graph):
    distributions = []

    for t_name in task_resources:
        resources = []
        for r_id in task_resources[t_name]:
            if r_id not in res_task_distr:
                continue

            distribution: DurationDistribution = res_task_distr[r_id][t_name]
            distribution_prosimos = distribution.to_prosimos_distribution()

            resources.append(
                {
                    "resource_id": r_id,
                    "distribution_name": distribution_prosimos["distribution_name"],
                    "distribution_params": distribution_prosimos["distribution_params"],
                }
            )

        distributions.append({"task_id": bpmn_graph.from_name[t_name], "resources": resources})

    return distributions


def _join_fuzzy_calendar_intervals(fuzzy_calendars, i_size):
    resource_calendars = []
    for r_id in fuzzy_calendars:
        resource_calendars.append(
            {
                "id": "%s_timetable" % r_id,
                "availability_probabilities": _sweep_line_intervals(fuzzy_calendars[r_id].res_absolute_prob, i_size),
                "workload_ratio": _sweep_line_intervals(fuzzy_calendars[r_id].res_relative_prob, i_size),
            }
        )
    return resource_calendars


def _sweep_line_intervals(prob_map, i_size):
    days_str = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY", 5: "SATURDAY", 6: "SUNDAY"}
    weekly_intervals = []
    for w_day in days_str:
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
        time_periods = []
        for from_i, to_i in joint_intervals:
            time_periods.append(
                {
                    "begin_time": str(_interval_index_to_time(from_i, i_size, True).time()),
                    "end_time": str(_interval_index_to_time(to_i, i_size, True).time()),
                    "probability": prob_map[w_day][from_i],
                }
            )
        weekly_intervals.append({"week_day": days_str[w_day], "fuzzy_intervals": time_periods})
    return weekly_intervals


def _interval_index_to_time(i_index, i_size, is_start):
    from_time = datetime.strptime("00:00:00", "%H:%M:%S") + timedelta(minutes=(i_index * i_size))
    return from_time if is_start else from_time + timedelta(minutes=i_size)


def _build_resource_profiles(p_info: ProcInfo):
    resource_profiles = []
    for t_name in p_info.task_resources:
        t_id = p_info.bpmn_graph.from_name[t_name]
        resource_list = []
        for r_id in p_info.task_resources[t_name]:
            if r_id not in p_info.fuzzy_calendars:
                continue
            resource_list.append(
                {
                    "id": r_id,
                    "name": r_id,
                    "cost_per_hour": 1,
                    "amount": 1,
                    "calendar": "%s_timetable" % r_id,
                    "assigned_tasks": [p_info.bpmn_graph.from_name[t_n] for t_n in p_info.resource_tasks[r_id]],
                }
            )
        resource_profiles.append({"id": t_id, "name": t_name, "resource_list": resource_list})
    return resource_profiles


def _distribution_to_json(distribution):
    distribution_params = []
    for d_param in distribution["distribution_params"]:
        distribution_params.append({"value": d_param})
    return {"distribution_name": distribution["distribution_name"], "distribution_params": distribution_params}


def _gateway_branching_to_json(gateways_branching):
    gateways_json = []
    for g_id in gateways_branching:
        probabilities = []
        g_prob = gateways_branching[g_id]
        for flow_arc in g_prob:
            probabilities.append({"path_id": flow_arc, "value": g_prob[flow_arc]})

        gateways_json.append({"gateway_id": g_id, "probabilities": probabilities})
    return gateways_json


def _check_probabilities_range(fuzzy_calendars):
    for r_id in fuzzy_calendars:
        i_fuzzy = fuzzy_calendars[r_id]
        for wd in i_fuzzy.res_relative_prob:
            for p in i_fuzzy.res_relative_prob[wd]:
                if p < 0 or p > 1:
                    print("Wrong Relative")
        for wd in i_fuzzy.res_absolute_prob:
            for p in i_fuzzy.res_absolute_prob[wd]:
                if p < 0 or p > 1:
                    print("Wrong Absolute")

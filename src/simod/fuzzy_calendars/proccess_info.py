from datetime import datetime, timedelta
from enum import Enum

import pytz
from prosimos.execution_info import TaskEvent


class Method(Enum):
    TRAPEZOIDAL = 0
    POINT = 1
    UNFIXED_INTERVALS = 2


class ProcInfo:
    def __init__(self, traces, i_size, with_negative_cases=True, method=Method.TRAPEZOIDAL, angle=1.0):
        self.traces = traces
        # self.bpmn_graph = bpmn_graph
        self.method = method
        self.angle = angle
        self.i_size = i_size
        self.max_interval_length = 0
        self.r_expected = dict()
        self.r_worked = dict()
        self.max_freq_i = self.init_weekly_intervals_count()
        self.total_intervals = self.init_weekly_intervals_count()
        self.from_date = pytz.utc.localize(datetime.max)
        self.to_date = pytz.utc.localize(datetime.min)
        self.r_t_events = dict()
        self.flow_arcs_frequency = dict()
        self.initial_events = dict()
        self.task_resources, self.resource_tasks = self.find_task_resource_assoc()
        self.allocation_prob = dict()
        self.res_busy = dict()
        self.compute_resource_frequencies(with_negative_cases, method)
        self.fuzzy_calendars = None

    def init_weekly_intervals_count(self):
        weekly_interval_freq = dict()
        for i in range(0, 7):
            weekly_interval_freq[i] = [0] * (1440 // self.i_size)
        return weekly_interval_freq

    def find_task_resource_assoc(self):
        task_resources = dict()
        resource_tasks = dict()
        for trace in self.traces:
            case_id = trace.p_case
            self.initial_events[case_id] = datetime(9999, 12, 31, tzinfo=pytz.UTC)
            # task_sequence = sort_by_completion_times(trace)
            # self.bpmn_graph.reply_trace(task_sequence, self.flow_arcs_frequency, True, trace.event_list)
            for ev in trace.event_list:
                self.initial_events[case_id] = min(self.initial_events[case_id], ev.started_at)
                if ev.resource_id not in resource_tasks:
                    resource_tasks[ev.resource_id] = set()
                    self.r_worked[ev.resource_id] = self.init_weekly_intervals_count()
                    self.r_expected[ev.resource_id] = self.init_weekly_intervals_count()
                    self.r_t_events[ev.resource_id] = dict()
                if ev.task_id not in task_resources:
                    task_resources[ev.task_id] = set()
                if ev.task_id not in self.r_t_events[ev.resource_id]:
                    self.r_t_events[ev.resource_id][ev.task_id] = list()
                task_resources[ev.task_id].add(ev.resource_id)
                resource_tasks[ev.resource_id].add(ev.task_id)
                self.r_t_events[ev.resource_id][ev.task_id].append(ev)

        return task_resources, resource_tasks

    def compute_resource_frequencies(self, with_negative_cases=True, method=Method.TRAPEZOIDAL):
        # max_waiting, max_processing = dict(), dict()
        self._compute_resource_busy_intervals()
        for trace in self.traces:
            trace.sort_by_completion_date(True)
            # if self.bpmn_graph is not None:
            #     compute_enabling_processing_times(trace, self.bpmn_graph, max_waiting, max_processing)
            for ev in trace.event_list:
                if method == Method.TRAPEZOIDAL:
                    if with_negative_cases and ev.enabled_at < ev.started_at:
                        self._trapezoidal_intervals(ev.enabled_at, ev.started_at, ev.resource_id, ev.task_id, False)
                    self._trapezoidal_intervals(ev.started_at, ev.completed_at, ev.resource_id, ev.task_id, True)
                elif method == Method.POINT and ev.enabled_at < ev.started_at:
                    if with_negative_cases:
                        self._observed_timestamps(ev.enabled_at, ev.started_at, ev.resource_id, ev.task_id, False)
                    self._observed_timestamps(ev.started_at, ev.completed_at, ev.resource_id, ev.task_id, True)
                self.update_interval_boundaries(ev)
        self._count_total_intervals_explored()

    def _compute_resource_busy_intervals(self):
        self.res_busy = dict()
        for trace in self.traces:
            for ev in trace.event_list:
                if ev.resource_id not in self.res_busy:
                    self.res_busy[ev.resource_id] = dict()
                self._check_busy_intervals(ev.started_at, ev.completed_at, self.res_busy[ev.resource_id])

    def _observed_timestamps(self, from_date, to_date, r_id, t_id, is_working):
        if not is_working:
            intervals = self.get_interval_indexes(from_date, to_date)
            for str_date, week_day, i in intervals:
                self._update_interval_frequency(r_id, t_id, str_date, week_day, i, is_working, 1.0)
        else:
            for c_date in [from_date, to_date]:
                i = (c_date.hour * 60 + c_date.minute) // self.i_size
                week_day = c_date.weekday()

                self._update_interval_frequency(r_id, t_id, str(c_date.date()), week_day, i, is_working, 1)

    def _trapezoidal_intervals(self, from_date, to_date, r_id, t_id, is_working):
        intervals = self.get_interval_indexes(from_date, to_date)
        self.max_interval_length = max(self.max_interval_length, len(intervals))
        if len(intervals) == 1:
            self._check_intervals(0, 0, intervals, r_id, t_id, is_working, 1.0)
        elif len(intervals) > 0:
            f_value = 1.0
            p_factor = 1.0
            if len(intervals) > 1:
                p_factor = 1.0 / ((len(intervals) // 2) * self.angle) if self.angle > 0 else 1.0

            s = 0
            e = len(intervals) - 1
            while f_value > 0:
                self._check_intervals(s, e, intervals, r_id, t_id, is_working, f_value)
                s += 1
                e -= 1
                f_value -= p_factor

    def _check_intervals(self, s, e, intervals, r_id, t_id, is_working, f_value):
        for x in [s, e]:
            str_date, week_day, i = intervals[x][0], intervals[x][1], intervals[x][2]
            self._update_interval_frequency(r_id, t_id, str_date, week_day, i, is_working, f_value)

    def get_interval_indexes(self, from_date, to_date):
        c_date = from_date
        intervals = list()

        while c_date <= to_date:
            str_date = str(c_date.date())
            from_minute, to_minute = _update_interval_boundaries(c_date, from_date, to_date)
            week_day = c_date.weekday()
            for i in range(int(from_minute / self.i_size), int(to_minute / self.i_size) + 1):
                intervals.append((str_date, week_day, i))

            c_date = (c_date + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        return intervals

    def _update_interval_frequency(self, r_id, t_id, str_date, week_day, i, is_working, f_value):
        for r_p in self.task_resources[t_id]:
            if str_date not in self.res_busy[r_id] or not self.res_busy[r_id][str_date][i]:
                self.r_expected[r_p][week_day][i] += 1
        if is_working:
            self.r_worked[r_id][week_day][i] += f_value
            self.r_expected[r_id][week_day][i] += 1
            self.max_freq_i[week_day][i] = max(self.max_freq_i[week_day][i], self.r_worked[r_id][week_day][i])

    def _count_total_intervals_explored(self):
        self.week_days_total = self.init_weekly_intervals_count()
        for d in range((self.to_date - self.from_date).days):
            w_day = (self.from_date + timedelta(days=d + 1)).weekday()
            for i in range(0, len(self.week_days_total[w_day])):
                self.week_days_total[w_day][i] += 1

    def _check_busy_intervals(self, from_date, to_date, dates_map):
        c_date = from_date
        while c_date <= to_date:
            from_minute, to_minute = _update_interval_boundaries(c_date, from_date, to_date)

            for i in range(int(from_minute / self.i_size), int(to_minute / self.i_size) + 1):
                self._visit_interval(c_date, i, dates_map)

            c_date = (c_date + timedelta(days=1)).replace(hour=0, minute=0, second=0)

    def _visit_interval(self, dt, i, dates_map):
        str_dt = str(dt.date())
        if str_dt not in dates_map:
            dates_map[str_dt] = [False] * (1440 // self.i_size)
        dates_map[str_dt][i] = True

    def update_interval_boundaries(self, evt_inf: TaskEvent):
        self.expand_interval_boundaries(evt_inf.enabled_at, evt_inf.completed_at)

    def expand_interval_boundaries(self, from_date, to_date):
        self.from_date = min(self.from_date, from_date)
        self.to_date = max(self.to_date, to_date)

    def get_index(self, from_date):
        return (from_date.hour * 60 + from_date.minute) // self.i_size

    def get_interval(self, i_index):
        return i_index * self.i_size, (i_index + 1) * self.i_size


def _update_interval_boundaries(c_date, from_date, to_date):
    from_minute = 0
    to_minute = 1439
    if c_date == from_date:
        from_minute = c_date.hour * 60 + c_date.minute
    if c_date.date() == to_date.date():
        to_minute = to_date.hour * 60 + to_date.minute
    return from_minute, to_minute


def compute_enabling_processing_times(trace_info, bpmn_graph, max_waiting, max_processing):
    flow_arcs_frequency = dict()
    task_sequence = list()
    for ev_info in trace_info.event_list:
        if ev_info.task_id not in max_waiting:
            max_waiting[ev_info.task_id] = 0
            max_processing[ev_info.task_id] = 0
        task_sequence.append(ev_info.task_id)

    _, _, _, enabling_times = bpmn_graph.reply_trace(task_sequence, flow_arcs_frequency, True, trace_info.event_list)
    for i in range(0, len(enabling_times)):
        ev_info = trace_info.event_list[i]
        if ev_info.started_at < enabling_times[i]:
            fix_enablement_from_incorrect_models(i, enabling_times, trace_info.event_list)
        ev_info.update_enabling_times(enabling_times[i])
        max_waiting[ev_info.task_id] = max(max_waiting[ev_info.task_id], ev_info.waiting_time)
        max_processing[ev_info.task_id] = max(max_processing[ev_info.task_id], ev_info.waiting_time)


def fix_enablement_from_incorrect_models(from_i: int, task_enablement: list, trace: list):
    started_at = trace[from_i].started_at
    enabled_at = task_enablement[from_i]
    i = from_i
    while i > 0:
        i -= 1
        if enabled_at == trace[i].completed_at:
            task_enablement[from_i] = started_at
            return True
    return False

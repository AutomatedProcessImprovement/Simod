from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import pytz
from pix_framework.io.event_log import EventLogIDs


class Method(Enum):
    TRAPEZOIDAL = 0
    POINT = 1
    UNFIXED_INTERVALS = 2


class Process:
    def __init__(
        self,
        i_size,
        activity_resources: dict[str, set[str]],  # {activity_id: set[resource_name]}
        log: pd.DataFrame,
        log_ids: EventLogIDs,
        with_negative_cases=True,
        method=Method.TRAPEZOIDAL,
        angle=1.0,
    ):
        self.log = log
        self.log_ids = log_ids
        self.method = method
        self.angle = angle
        self.i_size = i_size
        self.max_interval_length = 0
        self.r_expected = {}
        self.r_worked = {}
        self.max_freq_i = self.init_weekly_intervals_count()
        self.total_intervals = self.init_weekly_intervals_count()
        self.from_date = pytz.utc.localize(datetime.max)
        self.to_date = pytz.utc.localize(datetime.min)
        self.r_t_events = {}
        self.flow_arcs_frequency = {}
        self.initial_events = {}
        self.task_resources = activity_resources
        self._update_a_bunch_of_resource_related_internal_fields()  # position of this function call matters
        self.allocation_prob = {}
        self.res_busy = {}
        self.compute_resource_frequencies(with_negative_cases, method)
        self.fuzzy_calendars = None

    def init_weekly_intervals_count(self):
        weekly_interval_freq = {}
        for i in range(0, 7):
            weekly_interval_freq[i] = [0] * (1440 // self.i_size)
        return weekly_interval_freq

    def _update_a_bunch_of_resource_related_internal_fields(self):
        # initialize case start times
        for case_id, case_df in self.log.groupby(self.log_ids.case):
            # initial_events are the starting times of each case
            self.initial_events[case_id] = min(case_df[self.log_ids.start_time])

        # initialize resource related fields
        for resource_name in self.log[self.log_ids.resource].unique():
            self.r_worked[resource_name] = self.init_weekly_intervals_count()
            self.r_expected[resource_name] = self.init_weekly_intervals_count()
            self.r_t_events[resource_name] = {}
            resource_events = self.log[self.log[self.log_ids.resource] == resource_name]
            resource_activities = resource_events[self.log_ids.activity].unique()
            for activity_name in resource_activities:
                activity_events = resource_events[resource_events[self.log_ids.activity] == activity_name]
                activity_events = (
                    activity_events[[self.log_ids.start_time, self.log_ids.end_time, self.log_ids.enabled_time]]
                    .rename(
                        columns={
                            self.log_ids.start_time: "started_at",
                            self.log_ids.end_time: "completed_at",
                            self.log_ids.enabled_time: "enabled_at",
                        }
                    )
                    .to_dict("records")
                )  # fuzzy factory uses this format
                self.r_t_events[resource_name] = self.r_t_events[resource_name] | {activity_name: activity_events}

    def compute_resource_frequencies(self, with_negative_cases=True, method=Method.TRAPEZOIDAL):
        self._compute_resource_busy_intervals()

        for event in self.log.sort_values(by=self.log_ids.end_time).itertuples():
            enabled_at = getattr(event, self.log_ids.enabled_time)
            started_at = getattr(event, self.log_ids.start_time)
            completed_at = getattr(event, self.log_ids.end_time)
            resource_id = getattr(event, self.log_ids.resource)
            task_id = getattr(event, self.log_ids.activity)

            if method == Method.TRAPEZOIDAL:
                if with_negative_cases and enabled_at < started_at:
                    self._trapezoidal_intervals(enabled_at, started_at, resource_id, task_id, False)
                self._trapezoidal_intervals(started_at, completed_at, resource_id, task_id, True)
            elif method == Method.POINT and enabled_at < started_at:
                if with_negative_cases:
                    self._observed_timestamps(enabled_at, started_at, resource_id, task_id, False)
                self._observed_timestamps(started_at, completed_at, resource_id, task_id, True)

            self.update_interval_boundaries(enabled_at, completed_at)

        self._count_total_intervals_explored()

    def _compute_resource_busy_intervals(self):
        self.res_busy = {resource_name: {} for resource_name in self.log[self.log_ids.resource].unique()}

        for event in self.log[[self.log_ids.resource, self.log_ids.start_time, self.log_ids.end_time]].itertuples():
            self._check_busy_intervals(event.start_time, event.end_time, self.res_busy[event.resource])

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
        intervals = []

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

    def update_interval_boundaries(self, enabled_at, completed_at):
        self.expand_interval_boundaries(enabled_at, completed_at)

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

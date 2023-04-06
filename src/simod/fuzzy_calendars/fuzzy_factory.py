from numpy import percentile

from simod.discovery.distribution import get_best_distribution
from simod.fuzzy_calendars.intervals_frequency_calculator import ProcInfo


class IFuzzy:
    def __init__(self, p_info: ProcInfo):
        self.res_absolute_prob = p_info.init_weekly_intervals_count()
        self.res_relative_prob = p_info.init_weekly_intervals_count()


class FuzzyFactory:
    def __init__(self, freq_info: ProcInfo = None):
        self.freqs = freq_info

    def compute_resource_availability_calendars(self):
        freq = self.freqs
        f_calendars = dict()
        for r_id in freq.r_worked:
            r_f = IFuzzy(self.freqs)
            for wd in freq.r_worked[r_id]:
                for i in range(0, len(freq.r_worked[r_id][wd])):
                    if freq.r_worked[r_id][wd][i] > 0:
                        r_f.res_absolute_prob[wd][i] = freq.r_worked[r_id][wd][i] / freq.r_expected[r_id][wd][i]
                        r_f.res_relative_prob[wd][i] = freq.r_worked[r_id][wd][i] / freq.max_freq_i[wd][i]
            f_calendars[r_id] = r_f
        return f_calendars

    def compute_processing_times(self, fuzzy_calendars, min_obs=10, k=3):
        p_info = self.freqs
        filtered_events = self._filter_outliers(k)
        adjusted_durations, cumulative_durations = self._adjust_processing_times(filtered_events, fuzzy_calendars)
        res_task_distribution = dict()

        pending_distr = list()
        for r_id in adjusted_durations:
            res_task_distribution[r_id] = dict()
            for t_id in adjusted_durations[r_id]:
                res_task_distribution[r_id][t_id] = None
                if len(adjusted_durations[r_id][t_id]) >= min_obs:
                    res_task_distribution[r_id][t_id] = get_best_distribution(adjusted_durations[r_id][t_id])
                else:
                    pending_distr.append((r_id, t_id))

        for r_id, t_id in pending_distr:
            if res_task_distribution[r_id][t_id] is not None:
                continue
            best_candidate = None
            r_dur_mean = cumulative_durations[r_id][t_id] / len(adjusted_durations[r_id][t_id])
            closest_mean_diff = float("inf")
            for r_candidate in p_info.task_resources[t_id]:
                if res_task_distribution[r_candidate][t_id] is not None:
                    c_dur_mean = cumulative_durations[r_candidate][t_id] / len(adjusted_durations[r_candidate][t_id])
                    if abs(r_dur_mean - c_dur_mean) < closest_mean_diff:
                        best_candidate = r_candidate
                        closest_mean_diff = abs(r_dur_mean - c_dur_mean)
            if best_candidate is not None:
                res_task_distribution[r_id][t_id] = res_task_distribution[best_candidate][t_id]
            else:
                joint_durations = list()
                for r_candidate in p_info.task_resources[t_id]:
                    joint_durations.extend(adjusted_durations[r_candidate][t_id])
                joint_distribution = get_best_distribution(joint_durations)
                for r_joint in p_info.task_resources[t_id]:
                    res_task_distribution[r_joint][t_id] = joint_distribution
        return res_task_distribution

    def _filter_outliers(self, k):
        p_info = self.freqs
        filtered_events = dict()
        for r_id in p_info.r_t_events:
            filtered_events[r_id] = dict()
            for t_id in p_info.r_t_events[r_id]:
                events = p_info.r_t_events[r_id][t_id]
                data = self._get_event_durations(r_id, t_id)
                q25, q75 = percentile(data, 25), percentile(data, 75)
                iqr = q75 - q25
                # calculate the outlier cutoff
                cut_off = iqr * k
                lower, upper = q25 - cut_off, q75 + cut_off
                # identify outliers
                # outliers = [x for x in events if self._duration(x) < lower or self._duration(x) > upper]
                # remove outliers
                filtered_events[r_id][t_id] = [x for x in events if lower <= self._duration(x) <= upper]
        return filtered_events

    def _adjust_processing_times(self, filtered_events, fuzzy_calendars):
        p_info = self.freqs
        adjusted_durations = dict()
        cumulative_durations = dict()
        s_size = p_info.i_size * 60
        for r_id in filtered_events:
            adjusted_durations[r_id] = dict()
            cumulative_durations[r_id] = dict()
            for t_id in filtered_events[r_id]:
                adjusted_durations[r_id][t_id] = list()
                cumulative_durations[r_id][t_id] = 0
                for ev in filtered_events[r_id][t_id]:
                    intervals = p_info.get_interval_indexes(ev.started_at, ev.completed_at)
                    adj_dur = 0
                    n = len(intervals) - 1
                    for j in range(1, n):
                        (_, week_day, i) = intervals[j]
                        adj_dur += p_info.i_size * 60 * fuzzy_calendars[r_id].res_absolute_prob[week_day][i]

                    if n == 0:
                        adj_dur += (ev.completed_at - ev.started_at).total_seconds()
                    else:
                        i_0 = (s_size - self._diff_from_start(ev.started_at, intervals[0][2])) - ev.started_at.second
                        i_n = (self._diff_from_start(ev.completed_at, intervals[n][2])) + ev.completed_at.second

                        adj_dur += (i_0 * self._probability(r_id, fuzzy_calendars, intervals[0]))
                        adj_dur += (i_n * self._probability(r_id, fuzzy_calendars, intervals[n]))

                    adjusted_durations[r_id][t_id].append(adj_dur)
                    cumulative_durations[r_id][t_id] += adj_dur
        return adjusted_durations, cumulative_durations

    @staticmethod
    def _probability(r_id, fuzzy_calendar, interval):
        return fuzzy_calendar[r_id].res_absolute_prob[interval[1]][interval[2]]

    def _diff_from_start(self, b_date, i):
        diff_interval = list()
        i_start, _ = self.freqs.get_interval(i)
        for c_minute in [i_start, b_date.hour * 60 + b_date.minute]:
            diff_interval.append(c_minute * 60)
        return diff_interval[1] - diff_interval[0]

    def _get_event_durations(self, r_id, t_id):
        durations = list()
        for ev in self.freqs.r_t_events[r_id][t_id]:
            durations.append(self._duration(ev))
        return durations

    @staticmethod
    def _duration(ev):
        return (ev.completed_at - ev.started_at).total_seconds()
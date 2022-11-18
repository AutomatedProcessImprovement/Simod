from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List

import pandas as pd


class WeekDay(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @classmethod
    def from_string(cls, day_str: str):
        value_lowered = day_str.lower()
        if value_lowered == "monday":
            return WeekDay.MONDAY
        elif value_lowered == "tuesday":
            return WeekDay.TUESDAY
        elif value_lowered == "wednesday":
            return WeekDay.WEDNESDAY
        elif value_lowered == "thursday":
            return WeekDay.THURSDAY
        elif value_lowered == "friday":
            return WeekDay.FRIDAY
        elif value_lowered == "saturday":
            return WeekDay.SATURDAY
        elif value_lowered == "sunday":
            return WeekDay.SUNDAY

    def next(self):
        return WeekDay((self.value + 1) % 7)

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __repr__(self):
        return f'{self.value}'


@dataclass
class Interval:
    """
    Represents a general time interval without a specific date.
    """
    left_day: WeekDay
    right_day: WeekDay
    left_time: str  # e.g., "01:15:00" or "01:15:00.123456"
    right_time: str  # e.g., "02:15:00"
    _left_time: datetime.time
    _right_time: datetime.time

    def __init__(self, left_day: WeekDay, right_day: WeekDay, left_time: str, right_time: str):
        self.left_day = left_day
        self.right_day = right_day
        self.left_time = left_time
        self.right_time = right_time
        assert self.left_day == self.right_day, \
            f'Intervals must be on the same day, got: ' \
            f'{self.left_day} {self.left_time}, ' \
            f'{self.right_day} {self.right_time}'
        self._left_time = self.left_time_to_time()
        self._right_time = self.right_time_to_time()

    def __repr__(self):
        return f'(Interval({self.left_day} {self.left_time}, {self.right_day} {self.right_time}))'

    @staticmethod
    def _str_to_time(time_str: str) -> datetime.time:
        try:
            return datetime.strptime(time_str, '%H:%M:%S.%f').time()
        except ValueError:
            return datetime.strptime(time_str, '%H:%M:%S').time()

    def left_time_to_time(self) -> datetime.time:
        return Interval._str_to_time(self.left_time)

    def right_time_to_time(self) -> datetime.time:
        return Interval._str_to_time(self.right_time)

    def to_pd_interval(self) -> pd.Interval:
        # NOTE: year, month, day information must be discarded, only the time information is relevant.
        return pd.Interval(pd.Timestamp(self.left_time),
                           pd.Timestamp(self.right_time))

    def duration(self) -> pd.Timedelta:
        return pd.Timestamp(self.right_time) - pd.Timestamp(self.left_time)

    def overlaps(self, other: 'Interval') -> bool:
        days_overlap = self.left_day <= other.right_day and self.right_day >= other.left_day
        times_overlap = self.left_time_to_time() <= other.right_time_to_time() and self.right_time_to_time() >= other.left_time_to_time()
        return days_overlap and times_overlap

    def same_day_with(self, other: 'Interval') -> bool:
        return self.left_day == other.left_day and self.right_day == other.right_day

    def intersect(self, other: 'Interval') -> Optional['Interval']:
        if not self.overlaps(other):
            return None

        if self.same_day_with(other):
            intervals = sorted([self, other], key=lambda x: x._left_time)
            right_time = intervals[1].right_time \
                if intervals[0]._right_time > intervals[1]._right_time \
                else intervals[0].right_time

            intersection = Interval(left_day=intervals[0].left_day,
                                    right_day=intervals[0].right_day,
                                    left_time=intervals[1].left_time,
                                    right_time=right_time)

            if intersection.duration() == pd.Timedelta(0):
                return None
            return intersection

        else:
            raise NotImplementedError('Intervals must be on the same day')

    def within(self, other: 'Interval') -> bool:
        return self.left_within(other) and self.right_within(other)

    def left_within(self, other: 'Interval') -> bool:
        same_left_day = self.left_day == other.left_day
        left_time_within = other.left_time_to_time() <= self.left_time_to_time()
        return same_left_day and left_time_within

    def right_within(self, other: 'Interval') -> bool:
        same_right_day = self.right_day == other.right_day
        right_time_within = other.right_time_to_time() >= self.right_time_to_time()
        return same_right_day and right_time_within

    def subtract(self, other: 'Interval') -> List['Interval']:
        if not self.same_day_with(other):
            return [self]

        if not self.overlaps(other):
            return [self]

        day = self.left_day
        result = []

        if self._left_time < other._left_time:
            result.append(Interval(left_day=day, right_day=day, left_time=self.left_time, right_time=other.left_time))
        if self._right_time > other._right_time:
            result.append(Interval(left_day=day, right_day=day, left_time=other.right_time, right_time=self.right_time))

        return list(filter(lambda interval: interval.duration() > pd.Timedelta(0), result))


def overall_duration(intervals: [Interval]) -> pd.Timedelta:
    """
    Computes the overall duration of a set of intervals.
    """
    duration = pd.Timedelta(0)
    for interval in intervals:
        duration += interval.duration()
    return duration


def pd_interval_to_interval(interval: pd.Interval) -> [Interval]:
    """
    Recursive conversion of a pandas.Interval to a list of Interval, because a multiple-days interval is split into
    multiple single day intervals.
    """
    left_time = interval.left.time()
    left_day = interval.left.dayofweek
    right_time = interval.right.time()
    right_day = interval.right.dayofweek
    result = []
    if left_day == right_day:
        result.append(Interval(left_day=WeekDay(left_day),
                               right_day=WeekDay(right_day),
                               left_time=left_time.strftime('%H:%M:%S.%f'),
                               right_time=right_time.strftime('%H:%M:%S.%f')))
    else:
        result.append(
            Interval(left_day=WeekDay(left_day),
                     right_day=WeekDay(left_day),
                     left_time=left_time.strftime('%H:%M:%S.%f'),
                     right_time='23:59:59.999999')
        )

        new_left_of_the_rest = interval.left.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
        the_rest = pd.Interval(new_left_of_the_rest, interval.right)

        result.extend(pd_interval_to_interval(the_rest))

    return result


def pd_intervals_to_intervals(intervals: [pd.Interval]) -> [Interval]:
    """
    Converts a list of pandas.Interval to the list of Interval.
    """
    intervals = sorted(intervals, key=lambda item: item.left)
    result = []
    for interval in intervals:
        result.extend(pd_interval_to_interval(interval))
    return result


def prosimos_interval_to_interval(interval: dict) -> Interval:
    from_day = WeekDay.from_string(interval["from"])
    to_day = WeekDay.from_string(interval["to"])
    new_interval = Interval(left_day=from_day,
                            right_day=to_day,
                            left_time=interval["beginTime"],
                            right_time=interval["endTime"])
    return new_interval


def prosimos_interval_to_interval_safe(interval: dict) -> List[Interval]:
    """
    Converts a Prosimos interval to a list of custom intervals, and splits a multiple-days interval to multiple single
    day intervals.

    :param interval: Prosimos interval.
    :return: List of Interval.
    """

    from_day = WeekDay.from_string(interval["from"])
    to_day = WeekDay.from_string(interval["to"])

    if from_day == to_day:
        new_intervals = [Interval(left_day=from_day,
                                  right_day=to_day,
                                  left_time=interval["beginTime"],
                                  right_time=interval["endTime"])]
    else:
        new_intervals = [Interval(left_day=from_day,
                                  right_day=from_day,
                                  left_time=interval["beginTime"],
                                  right_time='23:59:59.999999')]
        return new_intervals + prosimos_interval_to_interval_safe({
            "from": from_day.next().name,
            "to": to_day.name,
            "beginTime": '00:00:00.000000',
            "endTime": interval["endTime"]
        })

    return new_intervals


def intersect_intervals(intervals1: [Interval], intervals2: [Interval]) -> [Interval]:
    """
    Computes the intersection of two sets of intervals.
    """
    result = []
    for interval1 in intervals1:
        for interval2 in intervals2:
            intersection = interval1.intersect(interval2)
            if intersection is not None:
                result.append(intersection)
    return result


def subtract_intervals(intervals1: [Interval], intervals2: [Interval]) -> [Interval]:
    """Subtracts intervals2 from intervals1."""

    if len(intervals1) == 0:
        return []

    if len(intervals2) == 0:
        return intervals1

    interval2 = intervals2[0]
    result = []
    for interval1 in intervals1:
        result.extend(interval1.subtract(interval2))

    if len(intervals2) == 1:
        return result

    result = subtract_intervals(result, intervals2[1:])

    return result


def remove_overlapping_time_from_intervals(a: List[Interval]) -> List[Interval]:
    """Removes overlapping time from intervals."""

    if len(a) == 0:
        return []

    if len(a) == 1:
        return a

    a = sorted(a, key=lambda x: x.to_pd_interval().left)

    result = subtract_intervals(a[1:], [a[0]])

    return [a[0]] + remove_overlapping_time_from_intervals(result)

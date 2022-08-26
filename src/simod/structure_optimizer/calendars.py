from dataclasses import dataclass
from enum import Enum
from typing import List


class WeekDay(Enum):
    MONDAY = 'MONDAY'
    TUESDAY = 'TUESDAY'
    WEDNESDAY = 'WEDNESDAY'
    THURSDAY = 'THURSDAY'
    FRIDAY = 'FRIDAY'
    SATURDAY = 'SATURDAY'
    SUNDAY = 'SUNDAY'


@dataclass
class Timetable:
    from_day: WeekDay
    to_day: WeekDay
    begin_time: str
    end_time: str

    @staticmethod
    def all_day_long() -> 'Timetable':
        return Timetable(
            from_day=WeekDay.MONDAY,
            to_day=WeekDay.SUNDAY,
            begin_time='00:00:00.000',
            end_time='23:59:59.999')

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'from': self.from_day.value,
            'to': self.to_day.value,
            'beginTime': self.begin_time,
            'endTime': self.end_time
        }


@dataclass
class Calendar:
    id: str
    name: str
    timetables: List[Timetable]

    @staticmethod
    def all_day_long() -> 'Calendar':
        return Calendar(
            id='24_7_CALENDAR',
            name='24_7_CALENDAR',
            timetables=[Timetable.all_day_long()])

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'id': self.id,
            'name': self.name,
            'time_periods': [timetable.to_dict() for timetable in self.timetables]
        }

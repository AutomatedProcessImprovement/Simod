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

    @staticmethod
    def work_hours() -> 'Timetable':
        return Timetable(
            from_day=WeekDay.MONDAY,
            to_day=WeekDay.SUNDAY,
            begin_time='09:00:00.000',
            end_time='17:00:00.000')

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'from': self.from_day.value,
            'to': self.to_day.value,
            'beginTime': self.begin_time,
            'endTime': self.end_time
        }

    @staticmethod
    def from_dict(d: dict) -> 'Timetable':
        return Timetable(
            from_day=WeekDay(d['from']),
            to_day=WeekDay(d['to']),
            begin_time=d['beginTime'],
            end_time=d['endTime']
        )

    @staticmethod
    def from_list_of_dicts(timetables: List[dict]) -> List['Timetable']:
        return [
            Timetable(
                from_day=WeekDay(d['from']),
                to_day=WeekDay(d['to']),
                begin_time=d['beginTime'],
                end_time=d['endTime']
            )
            for d in timetables
        ]


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

    @staticmethod
    def work_day() -> 'Calendar':
        return Calendar(
            id='9_5_CALENDAR',
            name='9_5_CALENDAR',
            timetables=[Timetable.work_hours()])

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'id': self.id,
            'name': self.name,
            'time_periods': [timetable.to_dict() for timetable in self.timetables]
        }

    @staticmethod
    def from_dict(calendar: dict) -> 'Calendar':
        return Calendar(
            id=calendar['id'],
            name=calendar['name'],
            timetables=[
                Timetable.from_dict(timetable) for timetable in calendar['time_periods']
            ]
        )

    def to_array(self) -> list:
        """For arrival calendars, Prosimos doesn't use 'id', 'name' and 'time_periods' keys, but accepts array of
        timetable dictionaries:"""
        return self.to_dict()['time_periods']

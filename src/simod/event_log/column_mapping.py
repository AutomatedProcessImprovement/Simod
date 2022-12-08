from dataclasses import dataclass, fields


@dataclass
class EventLogIDs:
    """Column mapping for the event log."""
    case: str = 'case_id'
    activity: str = 'activity'
    resource: str = 'resource'
    start_time: str = 'start_timestamp'
    end_time: str = 'end_timestamp'
    enabled_time: str = 'enabled_timestamp'
    processing_time: str = 'processing_time'
    available_time: str = 'available_time'
    estimated_start_time: str = 'estimated_start_time'
    role: str = 'role'

    @staticmethod
    def from_dict(config: dict) -> 'EventLogIDs':
        return EventLogIDs(**config)

    def to_dict(self) -> dict:
        return {
            attr.name: getattr(self, attr.name)
            for attr in fields(self.__class__)
        }

    def renaming_dict(self, to_ids: 'EventLogIDs') -> dict:
        return {
            getattr(self, attr.name): getattr(to_ids, attr.name)
            for attr in fields(self.__class__)
        }


SIMOD_DEFAULT_COLUMNS = EventLogIDs(
    case='caseid',
    activity='task',
    resource='user',
    start_time='start_timestamp',
    end_time='end_timestamp'
)

STANDARD_COLUMNS = EventLogIDs(
    case='case:concept:name',
    activity='concept:name',
    resource='org:resource',
    start_time='start_timestamp',
    end_time='time:timestamp'
)

PROSIMOS_COLUMNS = EventLogIDs(
    case='case_id',
    activity='activity',
    enabled_time='enable_time',
    start_time='start_time',
    end_time='end_time',
    resource='resource'
)

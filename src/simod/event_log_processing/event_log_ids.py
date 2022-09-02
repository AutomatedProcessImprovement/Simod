from dataclasses import dataclass


@dataclass
class EventLogIDs:
    """Column mapping for the event log."""
    case: str = 'case_id'
    activity: str = 'activity'
    resource: str = 'resource'
    start_time: str = 'start_timestamp'
    end_time: str = 'end_timestamp'
    enabled_time: str = 'enabled_timestamp'
    role: str = 'role'

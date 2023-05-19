from typing import List, Optional, Dict

import pandas as pd
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.discovery.calendar_factory import CalendarFactory
from pix_framework.log_ids import EventLogIDs

from simod.settings.temporal_settings import CalendarSettings, CalendarType
from simod.simulation.calendar_discovery.resource import full_day_schedule, working_hours_schedule
from simod.simulation.parameters.resource_profiles import ResourceProfile


def discover_resource_calendars_per_profile(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        calendar_settings: CalendarSettings,
        resource_profiles: List[ResourceProfile]
) -> Dict[str, RCalendar]:
    """
    Discover availability calendar for each resource profile in [resource_profiles]. If the
    calendar discovery type is 24/7 or 9/5, assign the corresponding calendar to each resource
    profile.

    When it is not possible to discover a resource calendar for a set of resource profiles, e.g.,
    lack of enough data, try:
     1 - Discover a single calendar for all the resource profiles with missing calendar.
     2 - If not, discover a single calendar for all the resource profiles in the log.
     3 - If not, assign the default 24/7 calendar.

    :param event_log: event log to discover the resource calendars from.
    :param log_ids: column IDs of the event log.
    :param calendar_settings: parameters for the calendar discovery.
    :param resource_profiles: list of resource profiles with their ID and resources.

    :return: list of availability calendars (one per profile).
    """
    calendar_type = calendar_settings.discovery_type
    if calendar_type == CalendarType.DEFAULT_24_7:
        # 24/7 calendar per resource profile
        resource_calendars = {}
        for resource_profile in resource_profiles:
            calendar_id = resource_profile.resources[0].calendar_id
            resource_calendars[calendar_id] = full_day_schedule(schedule_id=calendar_id)
    elif calendar_type == CalendarType.DEFAULT_9_5:
        # 9 to 5 calendar per resource profile
        resource_calendars = {}
        for resource_profile in resource_profiles:
            calendar_id = resource_profile.resources[0].calendar_id
            resource_calendars[calendar_id] = working_hours_schedule(schedule_id=calendar_id)
    elif calendar_type == CalendarType.UNDIFFERENTIATED:
        # Discover a resource calendar for all the resources in the log
        calendar_id = resource_profiles[0].resources[0].calendar_id
        resource_calendar = _discover_undifferentiated_resource_calendar(event_log, log_ids, calendar_settings)
        # Set discovered calendar, or default 24/7 if could not discover one
        resource_calendars = {}
        if resource_calendar is not None:
            resource_calendars[calendar_id] = resource_calendar
        else:
            resource_calendars[calendar_id] = full_day_schedule(schedule_id=calendar_id)
    else:
        # Discover a resource calendar per resource profile
        resource_calendars = _discover_resource_calendars_per_profile(event_log, log_ids, calendar_settings,
                                                                      resource_profiles)
    # Return discovered resource calendars
    return resource_calendars


def _discover_undifferentiated_resource_calendar(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        calendar_settings: CalendarSettings,
) -> Optional[RCalendar]:
    """
    Discover one availability calendar using all the timestamps in the received event log.

    :param event_log: event log to discover the resource calendar from.
    :param log_ids: column IDs of the event log.
    :param calendar_settings: parameters for the calendar discovery.
    :param calendar_id: ID to assign to the discovered calendar.

    :return: resource calendar for all the events in the received event log.
    """
    # Register each timestamp to the same profile
    calendar_factory = CalendarFactory(calendar_settings.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        activity = event[log_ids.activity]
        calendar_factory.check_date_time("Undifferentiated", activity, event[log_ids.start_time])
        calendar_factory.check_date_time("Undifferentiated", activity, event[log_ids.end_time])
    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        calendar_settings.confidence,
        calendar_settings.support,
        calendar_settings.participation
    )
    # Return resource calendar
    return discovered_timetables.get('Undifferentiated')


def _discover_resource_calendars_per_profile(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        calendar_settings: CalendarSettings,
        resource_profiles: List[ResourceProfile]
) -> Dict[str, RCalendar]:
    # Revert resource profiles
    resource_to_profile = {
        resource.id: resource_profile.id
        for resource_profile in resource_profiles
        for resource in resource_profile.resources
    }
    # Create map from each resource to its assigned calendar
    resources = [
        resource
        for resource_profile in resource_profiles
        for resource in resource_profile.resources
    ]
    resource_to_calendar_id = {
        resource.id: resource.calendar_id
        for resource in resources
    }

    # --- Discover a calendar per resource profile --- #

    # Register each timestamp to its corresponding profile
    calendar_factory = CalendarFactory(calendar_settings.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        profile_id = resource_to_profile[event[log_ids.resource]]
        activity = event[log_ids.activity]
        calendar_factory.check_date_time(profile_id, activity, event[log_ids.start_time])
        calendar_factory.check_date_time(profile_id, activity, event[log_ids.end_time])

    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        calendar_settings.confidence,
        calendar_settings.support,
        calendar_settings.participation
    )

    # Create calendar per resource profile
    resource_calendars = {}
    missing_resources = []
    for resource_id in resource_to_profile:
        if resource_id in discovered_timetables:
            resource_calendars[resource_id] = discovered_timetables[resource_id]
        else:
            missing_resources += [resource_id]

    # Check if there are resources with no calendars assigned
    if len(missing_resources) > 0:
        # Retain events performed by the resources with no calendar
        filtered_event_log = event_log[event_log[log_ids.resource].isin(missing_resources)]
        # Set common calendar id to missing resources
        calendar_id = "Grouped_resource_calendar"
        for resource in resources:
            if resource.id in missing_resources:
                resource.calendar_id = calendar_id
        # Discover one resource calendar for all of them
        resource_calendar = _discover_undifferentiated_resource_calendar(filtered_event_log, log_ids, calendar_settings)
        if resource_calendar is not None:
            for resource_name in missing_resources:
                resource_calendars[resource_name] = resource_calendar
        else:
            # Could not discover calendar for the missing resources, discover calendar with the entire log
            resource_calendar = _discover_undifferentiated_resource_calendar(event_log, log_ids, calendar_settings)
            if resource_calendar is not None:
                for resource_name in missing_resources:
                    resource_calendars[resource_name] = resource_calendar
            else:
                # Could not discover calendar for all the resources in the log, assign default 24/7
                calendar = full_day_schedule(schedule_id=calendar_id)
                for resource_name in missing_resources:
                    resource_calendars[resource_name] = calendar

    return resource_calendars

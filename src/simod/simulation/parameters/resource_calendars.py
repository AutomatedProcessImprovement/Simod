from typing import List, Optional

import pandas as pd
from pix_framework.log_ids import EventLogIDs
from prosimos.resource_calendar import CalendarFactory

from simod.settings.temporal_settings import CalendarType, CalendarDiscoveryParams
from simod.simulation.parameters.calendar import Calendar, Timetable
from simod.simulation.parameters.resource_profiles import ResourceProfile


def discover_resource_calendars_per_profile(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        params: CalendarDiscoveryParams,
        resource_profiles: List[ResourceProfile]
) -> List[Calendar]:
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
    :param params: parameters for the calendar discovery.
    :param resource_profiles: list of resource profiles with their ID and resources.

    :return: list of availability calendars (one per profile).
    """
    calendar_type = params.discovery_type
    if calendar_type == CalendarType.DEFAULT_24_7:
        # 24/7 calendar per resource profile
        resource_calendars = []
        for resource_profile in resource_profiles:
            calendar_id = resource_profile.resources[0].calendar_id
            resource_calendars += [
                Calendar(
                    id=calendar_id,
                    name=calendar_id,
                    timetables=[Timetable.all_day_long()]
                )
            ]
    elif calendar_type == CalendarType.DEFAULT_9_5:
        # 9 to 5 calendar per resource profile
        resource_calendars = []
        for resource_profile in resource_profiles:
            calendar_id = resource_profile.resources[0].calendar_id
            resource_calendars += [
                Calendar(
                    id=calendar_id,
                    name=calendar_id,
                    timetables=[Timetable.work_hours()]
                )
            ]
    elif calendar_type == CalendarType.UNDIFFERENTIATED:
        # Discover a resource calendar for all the resources in the log
        calendar_id = resource_profiles[0].resources[0].calendar_id
        resource_calendar = _discover_undifferentiated_resource_calendar(
            event_log,
            log_ids,
            params,
            calendar_id
        )
        # Set discovered calendar, or default 24/7 if could not discover one
        resource_calendars = [
            resource_calendar
            if resource_calendar is not None
            else Calendar(
                id=calendar_id,
                name=calendar_id,
                timetables=[Timetable.all_day_long()]
            )
        ]
    else:
        # Discover a resource calendar per resource profile
        resource_calendars = _discover_resource_calendars_per_profile(
            event_log,
            log_ids,
            params,
            resource_profiles
        )
    # Return discovered resource calendars
    return resource_calendars


def _discover_undifferentiated_resource_calendar(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        params: CalendarDiscoveryParams,
        calendar_id: str
) -> Optional[Calendar]:
    """
    Discover one availability calendar using all the timestamps in the received event log.

    :param event_log: event log to discover the resource calendar from.
    :param log_ids: column IDs of the event log.
    :param params: parameters for the calendar discovery.
    :param calendar_id: ID to assign to the discovered calendar.

    :return: resource calendar for all the events in the received event log.
    """
    # Register each timestamp to the same profile
    calendar_factory = CalendarFactory(params.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        activity = event[log_ids.activity]
        calendar_factory.check_date_time("Undifferentiated", activity, event[log_ids.start_time])
        calendar_factory.check_date_time("Undifferentiated", activity, event[log_ids.end_time])
    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        params.confidence,
        params.support,
        params.participation
    )
    # Return resource calendar
    return Calendar(
        id=calendar_id,
        name=calendar_id,
        timetables=Timetable.from_list_of_dicts(
            discovered_timetables['Undifferentiated'].to_json()
        )
    ) if 'Undifferentiated' in discovered_timetables else None


def _discover_resource_calendars_per_profile(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        params: CalendarDiscoveryParams,
        resource_profiles: List[ResourceProfile]
) -> List[Calendar]:
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
    calendar_factory = CalendarFactory(params.granularity)
    for _, event in event_log.iterrows():
        # Register start/end timestamps
        profile_id = resource_to_profile[event[log_ids.resource]]
        activity = event[log_ids.activity]
        calendar_factory.check_date_time(profile_id, activity, event[log_ids.start_time])
        calendar_factory.check_date_time(profile_id, activity, event[log_ids.end_time])
    # Discover weekly timetables
    discovered_timetables = calendar_factory.build_weekly_calendars(
        params.confidence,
        params.support,
        params.participation
    )
    # Create calendar per resource profile
    resource_calendars = []
    missing_resources = []
    for resource_id in resource_to_profile:
        if resource_id in discovered_timetables:
            timetable_dict = discovered_timetables[resource_id].to_json()
            resource_calendars += Calendar(
                id=resource_to_calendar_id[resource_id],
                name=resource_to_calendar_id[resource_id],
                timetables=Timetable.from_list_of_dicts(timetable_dict)
            )
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
        resource_calendar = _discover_undifferentiated_resource_calendar(
            filtered_event_log,
            log_ids,
            params,
            calendar_id
        )
        if resource_calendar is None:
            # Could not discover calendar for the missing resources, discover calendar with the entire log
            resource_calendar = _discover_undifferentiated_resource_calendar(
                event_log,
                log_ids,
                params,
                calendar_id
            )
            if resource_calendar is None:
                # Could not discover calendar for all the resources in the log, assign default 24/7
                resource_calendar = Calendar(
                    id=calendar_id,
                    name=calendar_id,
                    timetables=[Timetable.all_day_long()]
                )
        # Add grouped calendar to discovered resource calendars
        resource_calendars += [resource_calendar]
    # Return resource calendars
    return resource_calendars

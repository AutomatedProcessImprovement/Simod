from dataclasses import dataclass
from typing import List

import pandas as pd
from pix_framework.calendar.resource_calendar import RCalendar
from pix_framework.log_ids import EventLogIDs

from simod.settings.resource_model_settings import CalendarType, CalendarDiscoveryParams
from simod.simulation.parameters.resource_activity_performances import (
    ActivityResourceDistribution,
    discover_activity_resource_distribution,
)
from simod.simulation.parameters.resource_calendars import discover_resource_calendars_per_profile
from simod.simulation.parameters.resource_profiles import (
    ResourceProfile,
    discover_undifferentiated_resource_profile,
    discover_differentiated_resource_profiles,
    discover_pool_resource_profiles,
)


@dataclass
class ResourceModel:
    """
    Simulation model parameters containing the resource profiles, their calendars and their performance per activity.
    """

    resource_profiles: List[ResourceProfile]
    resource_calendars: List[RCalendar]
    activity_resource_distributions: List[ActivityResourceDistribution]

    def to_dict(self) -> dict:
        return {
            "resource_profiles": [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            "resource_calendars": [
                {
                    "id": calendar.calendar_id,
                    "name": calendar.calendar_id,
                    "time_periods": calendar.to_json(),
                }
                for calendar in self.resource_calendars
            ],
            "task_resource_distribution": [
                activity_resources.to_dict() for activity_resources in self.activity_resource_distributions
            ],
        }

    @staticmethod
    def from_dict(resource_model: dict) -> "ResourceModel":
        calendars = []
        for calendar_dict in resource_model["resource_calendars"]:
            calendar = RCalendar(calendar_id=calendar_dict["id"])
            for time_period in calendar_dict["time_periods"]:
                calendar.add_calendar_item(
                    from_day=time_period["from"],
                    to_day=time_period["to"],
                    begin_time=time_period["beginTime"],
                    end_time=time_period["endTime"],
                )
            calendars += [calendar]

        return ResourceModel(
            resource_profiles=[
                ResourceProfile.from_dict(resource_profile) for resource_profile in resource_model["resource_profiles"]
            ],
            resource_calendars=calendars,
            activity_resource_distributions=[
                ActivityResourceDistribution.from_dict(activity_resource_distribution)
                for activity_resource_distribution in resource_model["task_resource_distribution"]
            ],
        )


def discover_resource_model(
    event_log: pd.DataFrame, log_ids: EventLogIDs, params: CalendarDiscoveryParams
) -> ResourceModel:
    """
    Discover resource model parameters composed by the resource profiles, their calendars, and the resource-activity
    duration distributions.

    :param event_log: event log to discover the resource profiles, calendars, and performances from.
    :param log_ids: column IDs of the event log.
    :param params: parameters for the calendar discovery composed of the calendar type (default 24/7,
    default 9/5 undifferentiated, differentiates, or pools), and, if needed, the parameters for their discovery.

    :return: class with the resource profiles, their calendars, and the resource-activity duration distributions.
    """
    # --- Discover resource profiles --- #
    calendar_type = params.discovery_type
    if calendar_type in [CalendarType.DEFAULT_24_7, CalendarType.DEFAULT_9_5, CalendarType.UNDIFFERENTIATED]:
        resource_profiles = [discover_undifferentiated_resource_profile(event_log, log_ids)]
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_RESOURCE:
        resource_profiles = discover_differentiated_resource_profiles(event_log, log_ids)
    elif calendar_type == CalendarType.DIFFERENTIATED_BY_POOL:
        resource_profiles = discover_pool_resource_profiles(event_log, log_ids)
    else:
        raise ValueError(f"Unknown calendar discovery type: {calendar_type}")
    # Assert there are discovered resource profiles
    assert len(resource_profiles) > 0, "No resource profiles found"

    # --- Discover resource calendars for each profile --- #
    resource_calendars = discover_resource_calendars_per_profile(event_log, log_ids, params, resource_profiles)
    # Assert there are discovered resource calendars
    assert len(resource_calendars) > 0, "No resource calendars found"

    # --- Discover activity-resource performances given the resource profiles and calendars --- #
    activity_resource_distributions = discover_activity_resource_distribution(
        event_log, log_ids, resource_profiles, resource_calendars
    )
    # Assert there are discovered activity-resource performance
    assert len(activity_resource_distributions) > 0, "No activity resource distributions found"

    # --- Return resource model --- #
    return ResourceModel(
        resource_profiles=resource_profiles,
        resource_calendars=resource_calendars,
        activity_resource_distributions=activity_resource_distributions,
    )

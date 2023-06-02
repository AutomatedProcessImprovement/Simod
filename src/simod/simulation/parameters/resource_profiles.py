from dataclasses import dataclass, asdict
from typing import List

import pandas as pd
from pix_framework.discovery.resource_pools import discover_resource_pools
from pix_framework.log_ids import EventLogIDs


@dataclass
class Resource:
    """Simulation resource compatible with Prosimos."""

    id: str
    name: str
    amount: int
    cost_per_hour: float
    calendar_id: str
    assigned_tasks: List[str]

    @staticmethod
    def from_dict(resource: dict) -> "Resource":
        return Resource(
            id=resource["id"],
            name=resource["name"],
            amount=int(resource["amount"]),
            cost_per_hour=float(resource["cost_per_hour"]),
            calendar_id=resource["calendar"],
            assigned_tasks=resource["assignedTasks"],
        )


@dataclass
class ResourceProfile:
    """Simulation resource profile compatible with Prosimos."""

    id: str
    name: str
    resources: List[Resource]

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        result = asdict(self)

        # renaming for Prosimos
        result["resource_list"] = result.pop("resources")
        for resource in result["resource_list"]:
            resource["calendar"] = resource.pop("calendar_id")
            resource["assignedTasks"] = resource.pop("assigned_tasks")

        return result

    @staticmethod
    def from_dict(resource_profile: dict) -> "ResourceProfile":
        return ResourceProfile(
            id=resource_profile["id"],
            name=resource_profile["name"],
            resources=[Resource.from_dict(resource) for resource in resource_profile["resource_list"]],
        )


def discover_undifferentiated_resource_profile(
    event_log: pd.DataFrame,
    log_ids: EventLogIDs,
    calendar_id: str = "Undifferentiated_calendar",
    cost_per_hour: float = 20,
    keep_log_names: bool = True,
) -> "ResourceProfile":
    """
    Discover undifferentiated resource profile, by either keeping all the resource names observed in the event log
    as resources under that single profile, or by creating a single resource with the number of observed resources
    as amount.

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param calendar_id: ID of the calendar to assign to the created resource profile.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profile.
    :param keep_log_names: flag indicating if to summarize all the observed resources as one single resource with
    their number as the available amount (False), or create a resource per observed resource name (True).

    :return: resource profile with all the observed resources.
    """
    # All activities assigned to one single resource profile
    assigned_activities = list(event_log[log_ids.activity].unique())
    # Create resources for this profile
    if keep_log_names:
        # Create a resource for each resource name in the log
        resources = [
            Resource(
                id=name,
                name=name,
                amount=1,
                cost_per_hour=cost_per_hour,
                calendar_id=calendar_id,
                assigned_tasks=assigned_activities,
            )
            for name in event_log[log_ids.resource].unique()
        ]
    else:
        # Create a single resource with the number of different resources in the log
        resources = [
            Resource(
                id="Undifferentiated_resource",
                name="Undifferentiated_resource",
                amount=event_log[log_ids.resource].nunique(),
                cost_per_hour=cost_per_hour,
                calendar_id=calendar_id,
                assigned_tasks=assigned_activities,
            )
        ]
    # Return resource profile with all the single resources
    return ResourceProfile(
        id="Undifferentiated_resource_profile", name="Undifferentiated_resource_profile", resources=resources
    )


def discover_differentiated_resource_profiles(
    event_log: pd.DataFrame, log_ids: EventLogIDs, cost_per_hour: float = 20
) -> List["ResourceProfile"]:
    """
    Discover differentiated resource profiles (one resource profile per resource observed in the log).

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profiles.

    :return: list of resource profiles with all the observed resources.
    """
    # Create a profile for each discovered resource, with the activities they perform
    resource_profiles = []
    for resource_value, events in event_log.groupby(log_ids.resource):
        # Get list of performed activities
        resource_name = str(resource_value)
        assigned_activities = list(events[log_ids.activity].unique())
        # Create profile with default calendar ID
        resource_profiles += [
            ResourceProfile(
                id=f"{resource_name}_profile",
                name=f"{resource_name}_profile",
                resources=[
                    Resource(
                        id=resource_name,
                        name=resource_name,
                        amount=1,
                        cost_per_hour=cost_per_hour,
                        calendar_id=f"{resource_name}_calendar",
                        assigned_tasks=assigned_activities,
                    )
                ],
            )
        ]
    # Return list of profiles
    return resource_profiles


def discover_pool_resource_profiles(
    event_log: pd.DataFrame, log_ids: EventLogIDs, cost_per_hour: float = 20
) -> List["ResourceProfile"]:
    """
    Discover resource profiles grouped by pools. Discover pools of resources with the same characteristics, and
    create a resource profile per pool.

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profiles.

    :return: list of resource profiles with the observed resources grouped by pool.
    """
    pools = discover_resource_pools(event_log, log_ids)

    resource_profiles = []
    for pool_id in pools:
        # Get list of performed activities
        filtered_log = event_log[event_log[log_ids.resource].isin(pools[pool_id])]
        assigned_activities = list(filtered_log[log_ids.activity].unique())

        # Add resource profile with all the resources of this pool
        resource_profiles += [
            ResourceProfile(
                id=f"{pool_id}_profile",
                name=f"{pool_id}_profile",
                resources=[
                    Resource(
                        id=resource_name,
                        name=resource_name,
                        amount=1,
                        cost_per_hour=cost_per_hour,
                        calendar_id=f"{pool_id}_calendar",
                        assigned_tasks=assigned_activities,
                    )
                    for resource_name in pools[pool_id]
                ],
            )
        ]

    # Return resource profiles
    return resource_profiles

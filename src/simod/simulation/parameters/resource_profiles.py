from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from networkx import DiGraph
from pix_utils.log_ids import EventLogIDs

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.discovery.resource_pool_discoverer import ResourcePoolDiscoverer
from simod.simulation.calendar_discovery.resource import PoolMapping
from simod.simulation.parameters.calendar import Calendar


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
    def from_dict(resource: dict) -> 'Resource':
        return Resource(
            id=resource['id'],
            name=resource['name'],
            amount=int(resource['amount']),
            cost_per_hour=float(resource['cost_per_hour']),
            calendar_id=resource['calendar'],
            assigned_tasks=resource['assignedTasks']
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
        result['resource_list'] = result.pop('resources')
        for resource in result['resource_list']:
            resource['calendar'] = resource.pop('calendar_id')
            resource['assignedTasks'] = resource.pop('assigned_tasks')

        return result

    @staticmethod
    def from_dict(resource_profile: dict) -> 'ResourceProfile':
        return ResourceProfile(
            id=resource_profile['id'],
            name=resource_profile['name'],
            resources=[
                Resource.from_dict(resource)
                for resource in resource_profile['resource_list']
            ]
        )

    @staticmethod
    def undifferentiated(
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            bpmn_path: Path,
            calendar_id: str,
            resource_amount: Optional[int] = 1,
            total_number_of_resources: Optional[int] = None,
            cost_per_hour: float = 20) -> 'ResourceProfile':
        """Extracts undifferentiated resource profiles for Prosimos. For the structure optimizing stage,
        calendars do not matter.

        :param log: The event log to use.
        :param log_ids: The event log IDs to use.
        :param bpmn_path: The path to the BPMN model with activities and its IDs.
        :param calendar_id: The calendar ID that would be assigned to each resource.
        :param resource_amount: The amount of each distinct resource to use. NB: Prosimos has only 1 amount implemented at the moment.
        :param total_number_of_resources: The total amount of resources. If not specified, the number of resource is taken from the log
        :param cost_per_hour: The cost per hour of the resource.
        :return: The resource profile.
        """
        assigned_activities = [
            activity['task_id']
            for activity in BPMNReaderWriter(bpmn_path).read_activities()
        ]

        if total_number_of_resources is not None and total_number_of_resources > 0:
            resources_names = (f'SYSTEM_{i}' for i in range(total_number_of_resources))
        else:
            resources_names = log[log_ids.resource].unique().tolist()

        resources = [
            Resource(id=name,
                     name=name,
                     amount=resource_amount,
                     cost_per_hour=cost_per_hour,
                     calendar_id=calendar_id,
                     assigned_tasks=assigned_activities)
            for name in resources_names
        ]

        profile_name = 'UNDIFFERENTIATED_RESOURCE_PROFILE'
        profile = ResourceProfile(id=profile_name, name=profile_name, resources=list(resources))

        return profile

    @staticmethod
    def differentiated_by_pool_(
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            pool_by_resource_name: PoolMapping,
            process_graph: DiGraph,
            resource_amount: Optional[int] = 1,
            cost_per_hour: float = 20) -> List['ResourceProfile']:
        from simod.simulation.parameters.miner import get_activities_ids_by_name

        # NOTE: calendar.id == calendar.name == pool_name

        pool_key = 'pool'
        activity_id_key = 'activity_id'

        # Adding pool information to the log
        pool_data = pd.DataFrame(pool_by_resource_name.items(), columns=[log_ids.resource, pool_key])
        log = log.merge(pool_data, on=log_ids.resource)

        # Adding activity IDs to the log
        activity_ids = get_activities_ids_by_name(process_graph)
        activity_ids_data = pd.DataFrame(activity_ids.items(), columns=[log_ids.activity, activity_id_key])
        log = log.merge(activity_ids_data, on=log_ids.activity)

        # Finding activities' IDs for each pool
        activity_ids_by_pool = {
            pool_name: group[activity_id_key].unique().tolist()
            for (pool_name, group) in log.groupby(pool_key)
        }

        profiles = []
        for pool in log[pool_key].unique():
            activity_ids = activity_ids_by_pool[pool]
            pool_resources = log[log[pool_key] == pool][log_ids.resource].unique()

            resources = []
            for resource in pool_resources:
                cost = 0 if resource.lower() in ('start', 'end') else cost_per_hour
                resources.append(
                    Resource(
                        id=resource,
                        name=resource,
                        amount=resource_amount,
                        cost_per_hour=cost,
                        calendar_id=pool,
                        assigned_tasks=activity_ids
                    )
                )

            profiles.append(ResourceProfile(id=pool, name=pool, resources=resources))

        return profiles

    @staticmethod
    def differentiated_by_resource(
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            bpmn_path: Path,
            calendars: List[Calendar],
            resource_amount: Optional[int] = 1,
            cost_per_hour: float = 20) -> List['ResourceProfile']:

        # Activity names by resource name
        resource_names = log[log_ids.resource].unique()
        resource_activities = {resource_name: set() for resource_name in resource_names}
        for (resource_name, data) in log.groupby(log_ids.resource):
            activities = data[log_ids.activity].unique()
            resource_activities[resource_name] = set(activities)

        # Activities IDs mapping
        bpmn_reader = BPMNReaderWriter(bpmn_path)
        activity_ids_and_names = bpmn_reader.read_activities()

        # Calendars by resource name
        resource_calendars = {
            name: next(filter(lambda c, name_=name: c.name == name_, calendars))
            for name in resource_names
        }

        # Collecting profiles
        profiles = []
        for resource_name in resource_names:
            calendar = resource_calendars[resource_name]

            assigned_activities_ids = []
            for activity_name in resource_activities[resource_name]:
                try:
                    activity_id = next(filter(lambda a, name=activity_name: a['task_name'] == name,
                                              activity_ids_and_names))['task_id']
                    assigned_activities_ids.append(activity_id)
                except StopIteration:
                    raise ValueError(f'Activity {activity_name} is not found in the BPMN file')

            # NOTE: intervention to reduce cost for SYSTEM pool
            is_system_resource = resource_name.lower() == 'start' or resource_name.lower() == 'end'
            cost = 0 if is_system_resource else cost_per_hour
            # TODO: make sense to introduce Cost Structure Per Resource or Pool,
            #  so we have amount of resources and cost each resource

            resources = [
                Resource(id=resource_name,
                         name=resource_name,
                         amount=resource_amount,
                         cost_per_hour=cost,
                         calendar_id=calendar.id,
                         assigned_tasks=assigned_activities_ids)
            ]

            profiles.append(ResourceProfile(
                id=resource_name,
                name=resource_name,
                resources=resources
            ))

        return profiles


def discover_undifferentiated_resource_profile(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        activity_label_to_id: dict,
        calendar_id: str = "Undifferentiated_calendar",
        cost_per_hour: float = 20,
        keep_log_names: bool = True
) -> 'ResourceProfile':
    """
    Discover undifferentiated resource profile, by either keeping all the resource names observed in the event log
    as resources under that single profile, or by creating a single resource with the number of observed resources
    as amount.

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param calendar_id: ID of the calendar to assign to the created resource profile.
    :param activity_label_to_id: map with each activity label as key and its ID as value.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profile.
    :param keep_log_names: flag indicating if to summarize all the observed resources as one single resource with
    their number as the available amount (False), or create a resource per observed resource name (True).

    :return: resource profile with all the observed resources.
    """
    # All activities assigned to one single resource profile
    assigned_activities = list(activity_label_to_id.values())
    # Create resources for this profile
    if keep_log_names:
        # Create a resource for each resource name in the log
        resources = [
            Resource(
                id=name, name=name, amount=1, cost_per_hour=cost_per_hour,
                calendar_id=calendar_id, assigned_tasks=assigned_activities
            )
            for name in event_log[log_ids.resource].unique()
        ]
    else:
        # Create a single resource with the number of different resources in the log
        resources = [
            Resource(
                id="UNDIFFERENTIATED_RESOURCE", name="UNDIFFERENTIATED_RESOURCE",
                amount=event_log[log_ids.resource].nunique(), cost_per_hour=cost_per_hour,
                calendar_id=calendar_id, assigned_tasks=assigned_activities
            )
        ]
    # Return resource profile with all the single resources
    return ResourceProfile(
        id="UNDIFFERENTIATED_RESOURCE_PROFILE",
        name="UNDIFFERENTIATED_RESOURCE_PROFILE",
        resources=resources
    )


def discover_differentiated_resource_profiles(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        activity_label_to_id: dict,
        cost_per_hour: float = 20
) -> List['ResourceProfile']:
    """
    Discover differentiated resource profiles (one resource profile per resource observed in the log).

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param activity_label_to_id: map with each activity label as key and its ID as value.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profiles.

    :return: list of resource profiles with all the observed resources.
    """
    # Create a profile for each discovered resource, with the activities they perform
    resource_profiles = []
    for resource_value, events in event_log.groupby(log_ids.resource):
        # Get list of performed activities
        resource_name = str(resource_value)
        assigned_activities = [activity_label_to_id[activity_label] for activity_label in events[log_ids.activity].unique()]
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
                        assigned_tasks=assigned_activities
                    )
                ]
            )
        ]
    # Return list of profiles
    return resource_profiles


def discover_pool_resource_profiles(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        activity_label_to_id: dict,
        cost_per_hour: float = 20
) -> List['ResourceProfile']:
    """
    Discover resource profiles grouped by pools. Discover pools of resources with the same characteristics, and
    create a resource profile per pool.

    :param event_log: event log to discover the resource profiles from.
    :param log_ids: column IDs of the event log.
    :param activity_label_to_id: map with each activity label as key and its ID as value.
    :param cost_per_hour: cost per hour to assign to each resource in the current resource profiles.

    :return: list of resource profiles with the observed resources grouped by pool.
    """
    # Discover resource pools
    analyzer = ResourcePoolDiscoverer(
        event_log[[log_ids.activity, log_ids.resource]],
        activity_key=log_ids.activity,
        resource_key=log_ids.resource
    )
    # Map each pool ID to its resources
    pools = {}
    for item in analyzer.resource_table:
        pool_id = item['role']
        resources = pools.get(pool_id, [])
        resources += [item['resource']]
        pools[pool_id] = resources
    # Create profile for each pool
    resource_profiles = []
    for pool_id in pools:
        # Get list of performed activities
        filtered_log = event_log[event_log[log_ids.resource].isin(pools[pool_id])]
        assigned_activities = [activity_label_to_id[activity_label] for activity_label in filtered_log[log_ids.activity].unique()]
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
                        assigned_tasks=assigned_activities
                    )
                    for resource_name in pools[pool_id]
                ]
            )
        ]
    # Return resource profiles
    return resource_profiles

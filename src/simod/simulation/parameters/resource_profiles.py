from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from networkx import DiGraph

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.calendar_discovery.resource import PoolMapping
from simod.simulation.parameters.calendars import Calendar


@dataclass
class Resource:
    """Simulation resource compatible with Prosimos."""
    id: str
    name: str
    amount: int
    cost_per_hour: float
    calendar_id: Optional[str]
    assigned_tasks: Optional[List[str]] = None


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
    def differentiated_by_pool(
            log: pd.DataFrame,
            log_ids: EventLogIDs,
            bpmn_path: Path,
            calendars: List[Calendar],
            pool_mapping: PoolMapping,
            resource_amount: Optional[int] = 1,
            cost_per_hour: float = 20) -> List['ResourceProfile']:

        # Resource names per pool
        pool_names = list(set([pool_mapping[resource_name] for resource_name in pool_mapping]))
        resources_by_pool = {pool_name: set() for pool_name in pool_names}
        for resource_name in pool_mapping:
            resources_by_pool[pool_mapping[resource_name]].add(resource_name)

        # Activity names by resource name
        resource_names = log[log_ids.resource].unique()
        resource_activities = {resource_name: set() for resource_name in resource_names}
        for (resource_name, group) in log.groupby(log_ids.resource):
            activities = group[log_ids.activity].unique()
            resource_activities[resource_name] = set(activities)

        # Activities IDs mapping
        bpmn_reader = BPMNReaderWriter(bpmn_path)
        activity_ids_and_names = bpmn_reader.read_activities()
        activity_ids_by_name = {activity['task_name'].lower(): activity['task_id']
                                for activity in activity_ids_and_names}

        # Calendars by resource name
        resource_calendars = {
            name: next(filter(lambda c: c.name == resource, calendars))
            for name in pool_names
        }

        # Collecting profiles
        profiles = []
        for pool_name in pool_names:
            # TODO: introduce Cost Structure Per Resource or Pool
            cost = 0 if pool_name.lower() == 'system' else cost_per_hour
            calendar = resource_calendars[pool_name]
            resources = resources_by_pool[pool_name]

            assigned_activities_ids = []
            for resource in resources:
                for activity in resource_activities[resource]:
                    activity_id = activity_ids_by_name[activity.lower()]
                    assigned_activities_ids.append(activity_id)

            profiles.append(
                ResourceProfile(
                    id=pool_name,
                    name=pool_name,
                    resources=[
                        Resource(
                            id=pool_name,
                            name=pool_name,
                            amount=resource_amount,
                            cost_per_hour=cost,
                            calendar_id=calendar.id,
                            assigned_tasks=assigned_activities_ids
                        )
                    ]
                )
            )

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

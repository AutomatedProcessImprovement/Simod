from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

from simod.event_log import EventLogIDs
from simod.readers.bpmn_reader import BpmnReader


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
        """Extracts undifferentiated resource profiles for Prosimos. For the structure optimizing stage, calendars do not matter.

        :param log: The event log to use.
        :param log_ids: The event log IDs to use.
        :param bpmn_path: The path to the BPMN model with activities and its IDs.
        :param calendar_id: The calendar ID that would be assigned to each resource.
        :param resource_amount: The amount of each distinct resource to use. NB: Prosimos has only 1 amount implemented at the moment.
        :param total_number_of_resources: The total amount of resources. If not specified, the number of resource is taken from the log
        :param cost_per_hour: The cost per hour of the resource.

        Output must be able to be converted to the following JSON:
        {
            "resource_profiles": [
                {
                    "id": "Profile ID_1",
                    "name": "Credit Officer",
                    "resource_list": [
                        {
                          "id": "resource_id_1",
                          "name": "Credit Officer_1",
                          "cost_per_hour": "35",
                          "amount": 1,
                          "calendar": "sid-222A1118-4766-43B2-A004-7DADE521982D",
                          "assignedTasks": ["sid-622A1118-4766-43B2-A004-7DADE521982D"]
                        },
                    ]
                }
            ],
        }
        """

        # activities IDs
        activities_ids = []
        for activity in BpmnReader(bpmn_path).get_tasks_info():
            if activity['task_name'].lower() not in ['start', 'end']:
                activities_ids.append(activity['task_id'])

        # resource names
        if total_number_of_resources is not None and total_number_of_resources > 0:
            resources_names = (f'SYSTEM_{i}' for i in range(total_number_of_resources))
        else:
            resources_names = log[log_ids.resource].unique()

        # undifferentiated resources
        resources = [
            Resource(id=name,
                     name=name,
                     amount=resource_amount,
                     cost_per_hour=cost_per_hour,
                     calendar_id=calendar_id,
                     assigned_tasks=activities_ids)
            for name in resources_names
        ]

        profile_name = 'UNDIFFERENTIATED_RESOURCE_PROFILE'
        profile = ResourceProfile(id=profile_name, name=profile_name, resources=list(resources))

        return profile

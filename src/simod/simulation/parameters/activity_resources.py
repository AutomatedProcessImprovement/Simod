from dataclasses import dataclass
from typing import List


@dataclass
class ResourceDistribution:
    """Resource is the item of activity-resource duration distribution for Prosimos."""
    resource_id: str
    distribution: dict

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {'resource_id': self.resource_id} | self.distribution


@dataclass
class ActivityResourceDistribution:
    """Activity duration distribution per resource for Prosimos."""
    activity_id: str
    activity_resources_distributions: List[ResourceDistribution]

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'task_id': self.activity_id,
            'resources': [resource.to_dict() for resource in self.activity_resources_distributions]
        }

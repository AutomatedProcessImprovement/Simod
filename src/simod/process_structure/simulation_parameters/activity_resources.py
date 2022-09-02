from dataclasses import dataclass
from typing import List

from simod.process_structure.simulation_parameters.distributions import Distribution


@dataclass
class ResourceDistribution:
    """Resource is the item of activity-resource duration distribution for Prosimos."""
    resource_id: str
    distribution: Distribution

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {'resource_id': self.resource_id} | self.distribution.to_dict()


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

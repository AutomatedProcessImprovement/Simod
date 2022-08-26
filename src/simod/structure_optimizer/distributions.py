from dataclasses import dataclass
from enum import Enum
from typing import List


class DistributionKind(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'norm'
    TRIANGULAR = 'triang'
    EXPONENTIAL = 'expon'
    EXPONENTIAL_NORMAL = 'exponnorm'
    LOG_NORMAL = 'lognorm'
    GAMMA = 'gamma'

    @staticmethod
    def from_string(value: str) -> 'DistributionKind':
        name = value.lower()
        if name == 'uniform':
            return DistributionKind.UNIFORM
        elif name == 'norm':
            return DistributionKind.NORMAL
        elif name == 'triang':
            return DistributionKind.TRIANGULAR
        elif name == 'expon':
            return DistributionKind.EXPONENTIAL
        elif name == 'exponnorm':
            return DistributionKind.EXPONENTIAL_NORMAL
        elif name == 'lognorm':
            return DistributionKind.LOG_NORMAL
        elif name == 'gamma':
            return DistributionKind.GAMMA
        else:
            raise ValueError(f'Unknown distribution: {value}')


@dataclass
class DistributionParameter:
    value: float

    def to_dict(self) -> dict:
        return {
            'value': self.value
        }


@dataclass
class Distribution:
    kind: DistributionKind
    parameters: List[DistributionParameter]

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'distribution_name': self.kind.value,
            'distribution_params': [parameter.to_dict() for parameter in self.parameters]
        }

    @staticmethod
    def from_simod_dict(simod_dict: dict) -> 'Distribution':
        """Converts a dictionary from Simod to a Distribution object."""
        return Distribution(
            kind=DistributionKind(simod_dict['dname']),
            parameters=[DistributionParameter(value=parameter['value']) for parameter in simod_dict['dparams']]
        )
from dataclasses import dataclass
from enum import Enum
from typing import List

MAX_FLOAT = 1e+300


class DistributionType(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'norm'
    TRIANGULAR = 'triang'
    EXPONENTIAL = 'expon'
    LOG_NORMAL = 'lognorm'
    GAMMA = 'gamma'
    FIXED = 'fix'

    @staticmethod
    def from_string(value: str) -> 'DistributionType':
        name = value.lower()
        if name == 'uniform':
            return DistributionType.UNIFORM
        elif name in ('norm', 'normal'):
            return DistributionType.NORMAL
        elif name in ('triang', 'triangular'):
            return DistributionType.TRIANGULAR
        elif name in ('expon', 'exponential'):
            return DistributionType.EXPONENTIAL
        elif name in ('lognorm', 'log_normal', 'lognormal'):
            return DistributionType.LOG_NORMAL
        elif name == 'gamma':
            return DistributionType.GAMMA
        elif name in ['fix', 'fixed']:
            return DistributionType.FIXED
        else:
            raise ValueError(f'Unknown distribution: {value}')


class DistributionParameterType(Enum):
    LOC = 'loc'
    SCALE = 'scale'
    MIN = 'min'
    MAX = 'max'
    MODE = 'mode'
    SIGMA = 'sigma'
    SHAPE = 'shape'


@dataclass
class DistributionParameter:
    kind: DistributionParameterType
    value: float

    def to_dict(self) -> dict:
        """Prosimos accepts only 'value' key and infers the parameter type from its position in a list."""
        return {
            'value': self.value
        }


@dataclass
class Distribution:
    """Distribution class for activity-resource distributions for Prosimos. Prosimos accepts a list of at least
    4 parameters: loc, scale, min, max."""
    kind: DistributionType
    parameters: List[DistributionParameter]

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            'distribution_name': self.kind.value,
            'distribution_params': [parameter.to_dict() for parameter in self.parameters]
        }

    @staticmethod
    def fixed(value: float) -> dict:
        """Creates a fixed distribution with the given value."""
        return {
            'distribution_name': 'fix',
            'distribution_params': [
                {'value': value},
                {'value': value},
                {'value': value},
                {'value': value},
            ]
        }

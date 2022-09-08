from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np

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
    def from_simod_dict(simod_dict: dict, min_duration: float = 0, max_duration: float = MAX_FLOAT) -> 'Distribution':
        """Converts a dictionary from Simod to a Distribution object."""
        assert min_duration is not None and max_duration is not None, 'min_duration and max_duration must be provided'

        kind = DistributionType.from_string(simod_dict['dname'])

        if kind == DistributionType.UNIFORM:
            parameters = [
                DistributionParameter(DistributionParameterType.LOC, float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.SCALE,
                                      float(simod_dict['dparams']['arg2']) - float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.NORMAL:
            parameters = [
                DistributionParameter(DistributionParameterType.LOC, float(simod_dict['dparams']['mean'])),
                DistributionParameter(DistributionParameterType.SCALE, float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.TRIANGULAR:
            parameters = [
                DistributionParameter(DistributionParameterType.MODE, float(simod_dict['dparams']['mean'])),
                DistributionParameter(DistributionParameterType.LOC, float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.SCALE,
                                      float(simod_dict['dparams']['arg2']) - float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.EXPONENTIAL:
            parameters = [
                DistributionParameter(DistributionParameterType.LOC, 0),
                DistributionParameter(DistributionParameterType.SCALE, float(simod_dict['dparams']['arg1'])),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.LOG_NORMAL:
            mean_2 = float(simod_dict['dparams']['mean']) ** 2
            variance = float(simod_dict['dparams']['arg1'])
            phi = np.sqrt([variance + mean_2])[0]
            mu = np.log(mean_2 / phi)
            sigma = np.sqrt([np.log(phi ** 2 / mean_2)])[0]

            parameters = [
                DistributionParameter(DistributionParameterType.SIGMA, sigma),
                DistributionParameter(DistributionParameterType.LOC, 0),
                DistributionParameter(DistributionParameterType.SCALE, np.exp(mu)),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.GAMMA:
            mean, variance = float(simod_dict['dparams']['mean']), float(simod_dict['dparams']['arg1'])

            parameters = [
                DistributionParameter(DistributionParameterType.SHAPE, pow(mean, 2) / variance),
                DistributionParameter(DistributionParameterType.LOC, 0),
                DistributionParameter(DistributionParameterType.SCALE, variance / mean),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        elif kind == DistributionType.FIXED:
            parameters = [
                DistributionParameter(DistributionParameterType.LOC, float(simod_dict['dparams']['mean'])),
                DistributionParameter(DistributionParameterType.SCALE, 0),
                DistributionParameter(DistributionParameterType.MIN, min_duration),
                DistributionParameter(DistributionParameterType.MAX, max_duration)
            ]
        else:
            raise ValueError(f'Unknown distribution: {kind}')

        assert len(parameters) >= 4, 'Distribution must have at least 4 parameters for Prosimos'

        return Distribution(kind=kind, parameters=parameters)

    @staticmethod
    def fixed(value: float) -> 'Distribution':
        """Creates a fixed distribution with the given value."""
        return Distribution(kind=DistributionType.FIXED, parameters=[
            DistributionParameter(DistributionParameterType.LOC, value),
            DistributionParameter(DistributionParameterType.SCALE, 0),
            DistributionParameter(DistributionParameterType.MIN, value),
            DistributionParameter(DistributionParameterType.MAX, value)
        ])
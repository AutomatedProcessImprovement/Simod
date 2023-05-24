from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Tuple

from simod.settings.common_settings import Metric
from simod.utilities import parse_single_value_or_interval


class CalendarType(str, Enum):
    DEFAULT_24_7 = '24/7'  # 24/7 work day
    DEFAULT_9_5 = '9/5'  # 9 to 5 work day
    UNDIFFERENTIATED = 'undifferentiated'
    DIFFERENTIATED_BY_POOL = 'differentiated_by_pool'
    DIFFERENTIATED_BY_RESOURCE = 'differentiated_by_resource'

    @classmethod
    def from_str(cls, value: str) -> 'CalendarType':
        if value.lower() in ('default_24_7', 'dt247', '24_7', '247'):
            return cls.DEFAULT_24_7
        elif value.lower() in ('default_9_5', 'dt95', '9_5', '95'):
            return cls.DEFAULT_9_5
        elif value.lower() == 'undifferentiated':
            return cls.UNDIFFERENTIATED
        elif value.lower() in ('differentiated_by_pool', 'pool', 'pooled'):
            return cls.DIFFERENTIATED_BY_POOL
        elif value.lower() in ('differentiated_by_resource', 'differentiated'):
            return cls.DIFFERENTIATED_BY_RESOURCE
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == CalendarType.DEFAULT_24_7:
            return 'default_24_7'
        elif self == CalendarType.DEFAULT_9_5:
            return 'default_9_5'
        elif self == CalendarType.UNDIFFERENTIATED:
            return 'undifferentiated'
        elif self == CalendarType.DIFFERENTIATED_BY_POOL:
            return 'differentiated_by_pool'
        elif self == CalendarType.DIFFERENTIATED_BY_RESOURCE:
            return 'differentiated_by_resource'
        return f'Unknown CalendarType {str(self)}'


@dataclass
class CalendarDiscoveryParams:
    discovery_type: CalendarType = CalendarType.UNDIFFERENTIATED
    granularity: Optional[int] = 60  # minutes per granule
    confidence: Optional[float] = 0.1  # from 0 to 1.0
    support: Optional[float] = 0.1  # from 0 to 1.0
    participation: Optional[float] = 0.4  # from 0 to 1.0

    def to_dict(self) -> dict:
        # Save discovery type
        calendar_discovery_params = {
            'discovery_type': self.discovery_type.value
        }
        # Add calendar discovery parameters if any
        if self.discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL
        ]:
            calendar_discovery_params['granularity'] = self.granularity
            calendar_discovery_params['confidence'] = self.confidence
            calendar_discovery_params['support'] = self.support
            calendar_discovery_params['participation'] = self.participation
        # Return dict
        return calendar_discovery_params

    @staticmethod
    def from_dict(calendar_discovery_params: dict) -> 'CalendarDiscoveryParams':
        granularity = None
        confidence = None
        support = None
        participation = None
        # If the discovery type implies a discovery, parse parameters
        if calendar_discovery_params['discovery_type'] in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL
        ]:
            granularity = calendar_discovery_params['granularity']
            confidence = calendar_discovery_params['confidence']
            support = calendar_discovery_params['support']
            participation = calendar_discovery_params['participation']
        # Return parameters instance
        return CalendarDiscoveryParams(
            discovery_type=calendar_discovery_params['discovery_type'],
            granularity=granularity,
            confidence=confidence,
            support=support,
            participation=participation,
        )


@dataclass
class ResourceModelSettings:
    """
    Resource Model optimization settings.
    """
    optimization_metric: Metric = Metric.CIRCADIAN_EMD
    max_evaluations: int = 10
    num_evaluations_per_iteration: int = 3
    discovery_type: CalendarType = CalendarType.UNDIFFERENTIATED
    granularity: Optional[Union[int, Tuple[int, int]]] = (15, 60)  # minutes per granule
    confidence: Optional[Union[float, Tuple[float, float]]] = (0.5, 0.85)  # from 0 to 1.0
    support: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.3)  # from 0 to 1.0
    participation: Optional[Union[float, Tuple[float, float]]] = 0.4  # from 0 to 1.0

    @staticmethod
    def from_dict(config: dict) -> 'ResourceModelSettings':
        # Optimization metric
        optimization_metric = Metric.from_str(config.get('optimization_metric', "circadian_emd"))
        # Number of iterations for the optimization process
        max_evaluations = config.get('max_evaluations', 10)
        # Num evaluations per iteration
        num_evaluations_per_iteration = config.get('num_evaluations_per_iteration', 3)
        # Discovery type
        discovery_type = CalendarType.from_str(config.get('discovery_type', 'undifferentiated'))
        # Calendar discovery parameters
        if discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL
        ]:
            granularity = parse_single_value_or_interval(config.get('granularity', (15, 60)))
            confidence = parse_single_value_or_interval(config.get('confidence', (0.5, 0.85)))
            support = parse_single_value_or_interval(config.get('support', (0.01, 0.3)))
            participation = parse_single_value_or_interval(config.get('participation', 0.4))
        else:
            granularity, confidence, support, participation = None, None, None, None

        return ResourceModelSettings(
            optimization_metric=optimization_metric,
            max_evaluations=max_evaluations,
            num_evaluations_per_iteration=num_evaluations_per_iteration,
            discovery_type=discovery_type,
            granularity=granularity,
            confidence=confidence,
            support=support,
            participation=participation
        )

    def to_dict(self) -> dict:
        # Parse general settings
        dictionary = {
            'optimization_metric': self.optimization_metric.value,
            'max_evaluations': self.max_evaluations,
            'num_evaluations_per_iteration': self.num_evaluations_per_iteration,
            'discovery_type': self.discovery_type.value
        }
        # Parse calendar discovery parameters
        if self.discovery_type in [
            CalendarType.UNDIFFERENTIATED,
            CalendarType.DIFFERENTIATED_BY_RESOURCE,
            CalendarType.DIFFERENTIATED_BY_POOL
        ]:
            dictionary['granularity'] = self.granularity
            dictionary['confidence'] = self.confidence
            dictionary['support'] = self.support
            dictionary['participation'] = self.participation
        # Return dictionary
        return dictionary

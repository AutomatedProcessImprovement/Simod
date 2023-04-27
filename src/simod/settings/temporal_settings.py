from enum import Enum
from typing import Union, List, Optional, Tuple

from hyperopt import hp
from pydantic import BaseModel

from .common_settings import Metric


class CalendarType(str, Enum):
    DEFAULT_24_7 = '24/7'  # 24/7 work day
    DEFAULT_9_5 = '9/5'  # 9 to 5 work day
    UNDIFFERENTIATED = 'undifferentiated'
    DIFFERENTIATED_BY_POOL = 'differentiated_by_pool'
    DIFFERENTIATED_BY_RESOURCE = 'differentiated_by_resource'

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[CalendarType, List[CalendarType]]':
        if isinstance(value, str):
            return CalendarType._from_str(value)
        elif isinstance(value, int):
            return CalendarType._from_str(str(value))
        elif isinstance(value, list):
            return [CalendarType._from_str(v) for v in value]
        else:
            raise ValueError(f'Unknown value {value}')

    @classmethod
    def _from_str(cls, value: str) -> 'CalendarType':
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


class CalendarSettings(BaseModel):
    discovery_type: Union[CalendarType, List[CalendarType]]
    granularity: Optional[Union[int, List[int]]] = None  # minutes per granule
    confidence: Optional[Union[float, List[float]]] = None  # from 0 to 1.0
    support: Optional[Union[float, List[float]]] = None  # from 0 to 1.0
    participation: Optional[Union[float, List[float]]] = None  # from 0 to 1.0

    @staticmethod
    def default() -> 'CalendarSettings':
        """
        Default settings for calendar discovery. Used for case arrival rate discovery if no settings provided.
        """

        return CalendarSettings(
            discovery_type=CalendarType.UNDIFFERENTIATED,
            granularity=60,
            confidence=0.1,
            support=0.1,
            participation=0.4
        )

    @staticmethod
    def from_dict(config: dict) -> 'CalendarSettings':
        discovery_type = CalendarType.from_str(config.get('discovery_type', 'undifferentiated'))

        return CalendarSettings(
            discovery_type=discovery_type,
            granularity=config.get('granularity', 60),
            confidence=config.get('confidence', 0.1),
            support=config.get('support', 0.1),
            participation=config.get('participation', 0.4),
        )

    def to_hyperopt_options(self, prefix: str = '') -> List[tuple]:
        options = []

        discovery_types = self.discovery_type if isinstance(self.discovery_type, list) else [self.discovery_type]

        for dt in discovery_types:
            if dt in (CalendarType.UNDIFFERENTIATED, CalendarType.DIFFERENTIATED_BY_POOL,
                      CalendarType.DIFFERENTIATED_BY_RESOURCE):
                granularity = hp.uniform(f'{prefix}-{dt.name}-granularity', *self.granularity) \
                    if isinstance(self.granularity, list) \
                    else self.granularity
                confidence = hp.uniform(f'{prefix}-{dt.name}-confidence', *self.confidence) \
                    if isinstance(self.confidence, list) \
                    else self.confidence
                support = hp.uniform(f'{prefix}-{dt.name}-support', *self.support) \
                    if isinstance(self.support, list) \
                    else self.support
                participation = hp.uniform(f'{prefix}-{dt.name}-participation', *self.participation) \
                    if isinstance(self.participation, list) \
                    else self.participation
                options.append((dt.name,
                                {'granularity': granularity,
                                 'confidence': confidence,
                                 'support': support,
                                 'participation': participation}))
            else:
                # The rest options need only names because these are default calendars
                options.append((dt.name, {'calendar_type': dt.name}))

        return options

    @staticmethod
    def from_hyperopt_option(option: Tuple) -> 'CalendarSettings':
        calendar_type, calendar_parameters = option
        calendar_type = CalendarType.from_str(calendar_type)
        if calendar_type in (CalendarType.DEFAULT_9_5, CalendarType.DEFAULT_24_7):
            return CalendarSettings(discovery_type=calendar_type)
        else:
            return CalendarSettings(discovery_type=calendar_type, **calendar_parameters)

    def to_dict(self) -> dict:
        if isinstance(self.discovery_type, list):
            discovery_type = [dt.name for dt in self.discovery_type]
        else:
            discovery_type = self.discovery_type.name

        return {
            'discovery_type': discovery_type,
            'granularity': self.granularity,
            'confidence': self.confidence,
            'support': self.support,
            'participation': self.participation,
        }


class CalendarsSettings(BaseModel):
    optimization_metric: Metric
    max_evaluations: int
    case_arrival: Union[CalendarSettings, None]
    resource_profiles: Union[CalendarSettings, None]

    @staticmethod
    def default() -> 'CalendarsSettings':
        resource_settings = CalendarSettings.default()
        resource_settings.discovery_type = CalendarType.DIFFERENTIATED_BY_RESOURCE

        return CalendarsSettings(
            optimization_metric=Metric.ABSOLUTE_HOURLY_EMD,
            max_evaluations=1,
            case_arrival=CalendarSettings.default(),
            resource_profiles=resource_settings
        )

    @staticmethod
    def from_dict(config: dict) -> 'CalendarsSettings':
        # Case arrival is an optional parameter in the configuration file
        case_arrival = config.get('case_arrival')
        if case_arrival is not None:
            case_arrival = CalendarSettings.from_dict(case_arrival)
        else:
            case_arrival = CalendarSettings.default()

        resource_profiles = CalendarSettings.from_dict(config['resource_profiles'])

        optimization_metric = config.get('optimization_metric')
        if optimization_metric is not None:
            optimization_metric = Metric.from_str(optimization_metric)
        else:
            optimization_metric = Metric.ABSOLUTE_HOURLY_EMD

        return CalendarsSettings(
            optimization_metric=optimization_metric,
            max_evaluations=config['max_evaluations'],
            case_arrival=case_arrival,
            resource_profiles=resource_profiles,
        )

    def to_dict(self) -> dict:
        return {
            'optimization_metric': str(self.optimization_metric),
            'max_evaluations': self.max_evaluations,
            'case_arrival': self.case_arrival.to_dict(),
            'resource_profiles': self.resource_profiles.to_dict(),
        }

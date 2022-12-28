from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from simod.configuration import GatewayProbabilitiesDiscoveryMethod, CalendarType, CalendarSettings, \
    Configuration, Metric


@dataclass
class CalendarOptimizationSettings:
    """Settings for resources' and arrival calendars optimizer."""
    base_dir: Optional[Path]

    max_evaluations: int
    optimization_metric: Metric
    case_arrival: CalendarSettings
    resource_profiles: CalendarSettings

    simulation_repetitions: int = 1

    @staticmethod
    def from_configuration(config: Configuration, base_dir: Path) -> 'CalendarOptimizationSettings':
        return CalendarOptimizationSettings(
            base_dir=base_dir,
            optimization_metric=config.calendars.optimization_metric,
            simulation_repetitions=config.common.repetitions,
            max_evaluations=config.calendars.max_evaluations,
            case_arrival=config.calendars.case_arrival,
            resource_profiles=config.calendars.resource_profiles)

    def to_dict(self) -> dict:
        return {
            'base_dir': self.base_dir,
            'max_evaluations': self.max_evaluations,
            'optimization_metric': self.optimization_metric,
            'case_arrival': self.case_arrival.to_dict(),
            'resource_profiles': self.resource_profiles.to_dict(),
            'simulation_repetitions': self.simulation_repetitions,
        }


@dataclass
class ResourceOptimizationSettings:
    # in case of "discovered"
    res_confidence: Optional[float] = None
    res_support: Optional[float] = None

    # in case of "default"
    res_dtype: Optional[CalendarType] = None

    def __post_init__(self):
        assert (self.res_confidence is not None and self.res_support is not None) or (self.res_dtype is not None), \
            'Either resource confidence and support or calendar type should be specified'

    def to_dict(self) -> dict:
        return {
            'res_confidence': self.res_confidence if self.res_confidence is not None else None,
            'res_support': self.res_support if self.res_support is not None else None,
            'res_dtype': self.res_dtype.name if self.res_dtype is not None else None
        }


@dataclass
class ArrivalOptimizationSettings:
    # in case of "discovered"
    arr_confidence: Optional[float] = None
    arr_support: Optional[float] = None

    # in case of "default"
    arr_dtype: Optional[CalendarType] = None

    def __post_init__(self):
        assert (self.arr_confidence is not None and self.arr_support is not None) or (self.arr_dtype is not None), \
            'Either arrival confidence and support or calendar type should be specified'

    def to_dict(self) -> dict:
        return {
            'arr_confidence': self.arr_confidence if self.arr_confidence is not None else None,
            'arr_support': self.arr_support if self.arr_support is not None else None,
            'arr_dtype': self.arr_dtype.name if self.arr_dtype is not None else None
        }


class CalendarOptimizationType(Enum):
    """Type of optimization."""
    DISCOVERED = 1
    DEFAULT = 2

    @staticmethod
    def from_str(s: str) -> 'CalendarOptimizationType':
        if s.lower() == 'discovered':
            return CalendarOptimizationType.DISCOVERED
        elif s.lower() == 'default':
            return CalendarOptimizationType.DEFAULT
        else:
            raise ValueError(f'Unknown optimization type: {s}')

    def __str__(self):
        return self.name.lower()


@dataclass
class PipelineSettings:
    """Settings for the calendars optimizer pipeline."""
    # General settings
    output_dir: Path  # each pipeline run creates its own directory
    model_path: Path  # in calendars optimizer, this path doesn't change and just inherits from the project settings

    # Optimization settings

    # This one is taken from the structure settings, because it's not relevant to calendars
    # but is required for parameters extraction
    gateway_probabilities_method: Optional[GatewayProbabilitiesDiscoveryMethod]

    case_arrival: CalendarSettings
    resource_profiles: CalendarSettings

    @staticmethod
    def from_hyperopt_response(
            data: dict,
            initial_settings: CalendarOptimizationSettings,
            output_dir: Path,
            model_path: Path,
            gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod
    ) -> 'PipelineSettings':
        # Case arrival

        case_arrival_index = data['case_arrival']
        if isinstance(initial_settings.case_arrival.discovery_type, list):
            # if there's more than one type, use the index
            case_arrival_discovery_type = initial_settings.case_arrival.discovery_type[case_arrival_index]
        else:
            # otherwise, just take whatever is there
            case_arrival_discovery_type = initial_settings.case_arrival.discovery_type

        granularity = initial_settings.case_arrival.granularity
        confidence = initial_settings.case_arrival.confidence
        support = initial_settings.case_arrival.support
        participation = initial_settings.case_arrival.participation

        for (k, v) in data.items():
            # NOTE: 'case_arrival' is a prefix that has been passed to CalendarSettings.to_hyperopt_options
            # when hyperopt options were constructed
            if 'case_arrival' in k:
                if 'granularity' in k:
                    granularity = v
                if 'confidence' in k:
                    confidence = v
                if 'support' in k:
                    support = v
                if 'participation' in k:
                    participation = v

        case_arrival_settings = CalendarSettings(
            discovery_type=case_arrival_discovery_type,
            granularity=granularity,
            confidence=confidence,
            support=support,
            participation=participation,
        )

        # Resource profiles

        resource_profiles_index = data['resource_profiles']
        if isinstance(initial_settings.resource_profiles.discovery_type, list):
            resource_profiles_discovery_type = initial_settings.resource_profiles.discovery_type[
                resource_profiles_index]
        else:
            resource_profiles_discovery_type = initial_settings.resource_profiles.discovery_type

        granularity = initial_settings.resource_profiles.granularity
        confidence = initial_settings.resource_profiles.confidence
        support = initial_settings.resource_profiles.support
        participation = initial_settings.resource_profiles.participation

        for (k, v) in data.items():
            if 'resource_profile' in k:
                if 'granularity' in k:
                    granularity = v
                if 'confidence' in k:
                    confidence = v
                if 'support' in k:
                    support = v
                if 'participation' in k:
                    participation = v

        resource_profiles_settings = CalendarSettings(
            discovery_type=resource_profiles_discovery_type,
            granularity=granularity,
            confidence=confidence,
            support=support,
            participation=participation,
        )

        return PipelineSettings(
            gateway_probabilities_method=gateway_probabilities_method,
            output_dir=output_dir,
            model_path=model_path,
            case_arrival=case_arrival_settings,
            resource_profiles=resource_profiles_settings,
        )

    @staticmethod
    def from_hyperopt_option_dict(
            data: dict,
            output_dir: Path,
            model_path: Path,
            gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod) -> 'PipelineSettings':
        case_arrival_settings = CalendarSettings.from_hyperopt_option(data['case_arrival'])

        resource_profiles_settings = CalendarSettings.from_hyperopt_option(data['resource_profiles'])

        return PipelineSettings(
            gateway_probabilities_method=gateway_probabilities_method,
            output_dir=output_dir,
            model_path=model_path,
            case_arrival=case_arrival_settings,
            resource_profiles=resource_profiles_settings,
        )

    def to_dict(self) -> dict:
        return {
            'output_dir': str(self.output_dir),
            'model_path': str(self.model_path),
            'case_arrival': self.case_arrival.to_dict(),
            'resource_profiles': self.resource_profiles.to_dict(),
        }

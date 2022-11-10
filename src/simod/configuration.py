from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Tuple

from hyperopt import hp

from .event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS
from .utilities import get_project_dir

QBP_NAMESPACE_URI = 'http://www.qbp-simulator.com/Schema201212'
BPMN_NAMESPACE_URI = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
PROJECT_DIR = get_project_dir()


# Helper classes

class StructureMiningAlgorithm(Enum):
    SPLIT_MINER_1 = auto()
    SPLIT_MINER_2 = auto()
    SPLIT_MINER_3 = auto()

    @classmethod
    def from_str(cls, value: str) -> 'StructureMiningAlgorithm':
        if value.lower() in ['sm1', 'splitminer1', 'split miner 1', 'split_miner_1', 'split-miner-1']:
            return cls.SPLIT_MINER_1
        elif value.lower() in ['sm2', 'splitminer2', 'split miner 2', 'split_miner_2', 'split-miner-2']:
            return cls.SPLIT_MINER_2
        elif value.lower() in ['sm3', 'splitminer3', 'split miner 3', 'split_miner_3', 'split-miner-3']:
            return cls.SPLIT_MINER_3
        else:
            raise ValueError(f'Unknown structure mining algorithm: {value}')

    def __str__(self):
        if self == StructureMiningAlgorithm.SPLIT_MINER_1:
            return 'Split Miner 1'
        elif self == StructureMiningAlgorithm.SPLIT_MINER_2:
            return 'Split Miner 2'
        elif self == StructureMiningAlgorithm.SPLIT_MINER_3:
            return 'Split Miner 3'
        return f'Unknown StructureMiningAlgorithm {str(self)}'


class GatewayProbabilitiesDiscoveryMethod(Enum):
    DISCOVERY = auto()
    EQUIPROBABLE = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[GatewayProbabilitiesDiscoveryMethod, ' \
                                                       'List[GatewayProbabilitiesDiscoveryMethod]]':
        if isinstance(value, str):
            return GatewayProbabilitiesDiscoveryMethod._from_str(value)
        elif isinstance(value, list):
            return [GatewayProbabilitiesDiscoveryMethod._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'GatewayProbabilitiesDiscoveryMethod':
        if value.lower() == 'discovery':
            return cls.DISCOVERY
        elif value.lower() == 'equiprobable':
            return cls.EQUIPROBABLE
        elif value.lower() == 'random':
            return cls.RANDOM
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == GatewayProbabilitiesDiscoveryMethod.DISCOVERY:
            return 'discovery'
        elif self == GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE:
            return 'equiprobable'
        elif self == GatewayProbabilitiesDiscoveryMethod.RANDOM:
            return 'random'
        return f'Unknown GateManagement {str(self)}'


class CalendarType(Enum):
    DEFAULT_24_7 = auto()  # 24/7 work day
    DEFAULT_9_5 = auto()  # 9 to 5 work day
    UNDIFFERENTIATED = auto()
    DIFFERENTIATED_BY_POOL = auto()
    DIFFERENTIATED_BY_RESOURCE = auto()

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


class Metric(Enum):
    TSD = auto()
    DAY_HOUR_EMD = auto()
    LOG_MAE = auto()
    DL = auto()
    MAE = auto()
    DAY_EMD = auto()
    CAL_EMD = auto()
    DL_MAE = auto()
    HOUR_EMD = auto()
    ABSOLUTE_HOURLY_EMD = auto()
    CYCLE_TIME_EMD = auto()

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[Metric, List[Metric]]':
        if isinstance(value, str):
            return Metric._from_str(value)
        elif isinstance(value, list):
            return [Metric._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'Metric':
        if value.lower() == 'tsd':
            return cls.TSD
        elif value.lower() == 'day_hour_emd':
            return cls.DAY_HOUR_EMD
        elif value.lower() == 'log_mae':
            return cls.LOG_MAE
        elif value.lower() == 'dl':
            return cls.DL
        elif value.lower() == 'mae':
            return cls.MAE
        elif value.lower() == 'day_emd':
            return cls.DAY_EMD
        elif value.lower() == 'cal_emd':
            return cls.CAL_EMD
        elif value.lower() == 'dl_mae':
            return cls.DL_MAE
        elif value.lower() == 'hour_emd':
            return cls.HOUR_EMD
        elif value.lower() in ('absolute_hourly_emd', 'absolute_hour_emd', 'abs_hourly_emd', 'abs_hour_emd'):
            return cls.ABSOLUTE_HOURLY_EMD
        elif value.lower() == 'cycle_time_emd':
            return cls.CYCLE_TIME_EMD
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == Metric.TSD:
            return 'TSD'
        elif self == Metric.DAY_HOUR_EMD:
            return 'DAY_HOUR_EMD'
        elif self == Metric.LOG_MAE:
            return 'LOG_MAE'
        elif self == Metric.DL:
            return 'DL'
        elif self == Metric.MAE:
            return 'MAE'
        elif self == Metric.DAY_EMD:
            return 'DAY_EMD'
        elif self == Metric.CAL_EMD:
            return 'CAL_EMD'
        elif self == Metric.DL_MAE:
            return 'DL_MAE'
        elif self == Metric.HOUR_EMD:
            return 'HOUR_EMD'
        elif self == Metric.ABSOLUTE_HOURLY_EMD:
            return 'ABSOLUTE_HOURLY_EMD'
        elif self == Metric.CYCLE_TIME_EMD:
            return 'CYCLE_TIME_EMD'
        return f'Unknown Metric {str(self)}'


class PDFMethod(Enum):
    AUTOMATIC = auto()
    SEMIAUTOMATIC = auto()
    MANUAL = auto()
    DEFAULT = auto()

    @classmethod
    def from_str(cls, value: str) -> 'PDFMethod':
        if value.lower() == 'automatic':
            return cls.AUTOMATIC
        elif value.lower() == 'semiautomatic':
            return cls.SEMIAUTOMATIC
        elif value.lower() == 'manual':
            return cls.MANUAL
        elif value.lower() == 'default':
            return cls.DEFAULT
        else:
            raise ValueError(f'Unknown value {value}')


class ExecutionMode(Enum):
    SINGLE = auto()
    OPTIMIZER = auto()

    @classmethod
    def from_str(cls, value: str) -> 'ExecutionMode':
        if value.lower() == 'single':
            return cls.SINGLE
        elif value.lower() == 'optimizer':
            return cls.OPTIMIZER
        else:
            raise ValueError(f'Unknown value {value}')


# Settings classes

@dataclass
class CommonSettings:
    log_path: Path
    log_ids: Optional[EventLogIDs]
    model_path: Optional[Path]
    exec_mode: ExecutionMode
    repetitions: int
    simulation: bool
    evaluation_metrics: Union[Metric, List[Metric]]
    clean_intermediate_files: bool

    @staticmethod
    def from_dict(config: dict) -> 'CommonSettings':
        log_path = Path(config['log_path'])
        if not log_path.is_absolute():
            log_path = PROJECT_DIR / log_path

        exec_mode = ExecutionMode.from_str(config['exec_mode'])

        metrics = [Metric.from_str(metric) for metric in config['evaluation_metrics']]

        mapping = config.get('log_ids', None)
        if mapping is not None:
            log_ids = EventLogIDs.from_dict(mapping)
        else:
            log_ids = STANDARD_COLUMNS

        clean_up = config.get('clean_intermediate_files', False)

        model_path = config.get('model_path', None)
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_DIR / model_path

        return CommonSettings(
            log_path=log_path,
            log_ids=log_ids,
            model_path=model_path,
            exec_mode=exec_mode,
            repetitions=config['repetitions'],
            simulation=config['simulation'],
            evaluation_metrics=metrics,
            clean_intermediate_files=clean_up
        )


@dataclass
class PreprocessingSettings:
    multitasking: bool

    @staticmethod
    def from_dict(config: dict) -> 'PreprocessingSettings':
        return PreprocessingSettings(
            multitasking=config['multitasking'],
        )


@dataclass
class StructureSettings:
    # Flag to turn off structure discovery, the model must be provided in CommonSettings then
    disable_discovery: bool

    # Structure discovery settings

    optimization_metric: Metric = Metric.DL
    max_evaluations: Optional[int] = None
    mining_algorithm: Optional[StructureMiningAlgorithm] = None
    concurrency: Optional[Union[float, List[float]]] = None
    epsilon: Optional[Union[float, List[float]]] = None
    eta: Optional[Union[float, List[float]]] = None
    gateway_probabilities: Optional[
        Union[GatewayProbabilitiesDiscoveryMethod, List[GatewayProbabilitiesDiscoveryMethod]]] = None
    or_rep: Optional[Union[bool, List[bool]]] = None
    and_prior: Optional[Union[bool, List[bool]]] = None
    distribution_discovery_type: Optional[PDFMethod] = None

    @staticmethod
    def from_dict(config: dict) -> 'StructureSettings':
        disable_discovery = config.get('disable_discovery', False)
        if disable_discovery:
            return StructureSettings(disable_discovery=True)

        mining_algorithm = StructureMiningAlgorithm.from_str(config['mining_algorithm'])

        gateway_probabilities = [GatewayProbabilitiesDiscoveryMethod.from_str(g) for g in
                                 config['gateway_probabilities']]

        dst = config.get('distribution_discovery_type')
        if dst is not None:
            distribution_discovery_type = PDFMethod.from_str(dst)
        else:
            distribution_discovery_type = PDFMethod.AUTOMATIC

        optimization_metric = config.get('optimization_metric')
        if optimization_metric is not None:
            optimization_metric = Metric.from_str(optimization_metric)
        else:
            optimization_metric = Metric.DL

        return StructureSettings(
            disable_discovery=disable_discovery,
            optimization_metric=optimization_metric,
            max_evaluations=config['max_evaluations'],
            mining_algorithm=mining_algorithm,
            concurrency=config['concurrency'],
            epsilon=config['epsilon'],
            eta=config['eta'],
            gateway_probabilities=gateway_probabilities,
            or_rep=config['or_rep'],
            and_prior=config['and_prior'],
            distribution_discovery_type=distribution_discovery_type
        )


@dataclass
class CalendarSettings:
    discovery_type: Union[CalendarType, List[CalendarType]]
    granularity: Optional[Union[int, List[int]]] = None  # minutes per granule
    confidence: Optional[Union[float, List[float]]] = None  # from 0 to 1.0
    support: Optional[Union[float, List[float]]] = None  # from 0 to 1.0
    participation: Optional[Union[float, List[float]]] = None  # from 0 to 1.0

    @staticmethod
    def default() -> 'CalendarSettings':
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


@dataclass
class CalendarsSettings:
    optimization_metric: Metric
    max_evaluations: int
    case_arrival: CalendarSettings
    resource_profiles: CalendarSettings

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


@dataclass
class Configuration:
    common: CommonSettings
    preprocessing: PreprocessingSettings
    structure: StructureSettings
    calendars: CalendarsSettings

    @staticmethod
    def from_yaml(config: dict) -> 'Configuration':
        assert config['version'] == 2, 'Configuration version must be 2'

        common_settings = CommonSettings.from_dict(config['common'])
        preprocessing_settings = PreprocessingSettings.from_dict(config['preprocessing'])
        structure_settings = StructureSettings.from_dict(config['structure'])
        calendars_settings = CalendarsSettings.from_dict(config['calendars'])

        return Configuration(
            common=common_settings,
            preprocessing=preprocessing_settings,
            structure=structure_settings,
            calendars=calendars_settings,
        )

    @staticmethod
    def from_stream(stream) -> 'Configuration':
        import yaml
        config = yaml.safe_load(stream)
        return Configuration.from_yaml(config)

    @staticmethod
    def from_path(file_path: Path) -> 'Configuration':
        with file_path.open() as f:
            return Configuration.from_stream(f)

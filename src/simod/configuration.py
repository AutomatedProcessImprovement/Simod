from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Tuple

import yaml
from hyperopt import hp

from .cli_formatter import print_notice
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
            raise NotImplementedError('Random gateway probabilities are not supported')
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
    DL = auto()
    CIRCADIAN_EMD = auto()
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
        if value.lower() == 'dl':
            return cls.DL
        elif value.lower() == 'circadian_emd':
            return cls.CIRCADIAN_EMD
        elif value.lower() in ('absolute_hourly_emd', 'absolute_hour_emd', 'abs_hourly_emd', 'abs_hour_emd'):
            return cls.ABSOLUTE_HOURLY_EMD
        elif value.lower() == 'cycle_time_emd':
            return cls.CYCLE_TIME_EMD
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == Metric.DL:
            return 'DL'
        elif self == Metric.CIRCADIAN_EMD:
            return 'CIRCADIAN_EMD'
        elif self == Metric.ABSOLUTE_HOURLY_EMD:
            return 'ABSOLUTE_HOURLY_EMD'
        elif self == Metric.CYCLE_TIME_EMD:
            return 'CYCLE_TIME_EMD'
        return f'Unknown Metric {str(self)}'


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
    test_log_path: Optional[Path]
    log_ids: Optional[EventLogIDs]
    model_path: Optional[Path]
    exec_mode: ExecutionMode
    repetitions: int
    evaluation_metrics: Union[Metric, List[Metric]]
    clean_intermediate_files: bool
    extraneous_activity_delays: bool

    @staticmethod
    def default() -> 'CommonSettings':
        return CommonSettings(
            log_path=Path('example_log.csv'),
            test_log_path=None,
            log_ids=STANDARD_COLUMNS,
            model_path=None,
            exec_mode=ExecutionMode.OPTIMIZER,
            repetitions=1,
            evaluation_metrics=[Metric.DL, Metric.ABSOLUTE_HOURLY_EMD, Metric.CIRCADIAN_EMD, Metric.CYCLE_TIME_EMD],
            clean_intermediate_files=False,
            extraneous_activity_delays=False,
        )

    @staticmethod
    def from_dict(config: dict) -> 'CommonSettings':
        log_path = Path(config['log_path'])
        if not log_path.is_absolute():
            log_path = PROJECT_DIR / log_path

        test_log_path = config.get('test_log_path', None)
        if test_log_path is not None:
            test_log_path = Path(test_log_path)
            if not test_log_path.is_absolute():
                test_log_path = PROJECT_DIR / test_log_path

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

        extraneous_activity_delays = config.get('extraneous_activity_delays', False)

        return CommonSettings(
            log_path=log_path,
            test_log_path=test_log_path,
            log_ids=log_ids,
            model_path=model_path,
            exec_mode=exec_mode,
            repetitions=config['repetitions'],
            evaluation_metrics=metrics,
            clean_intermediate_files=clean_up,
            extraneous_activity_delays=extraneous_activity_delays,
        )

    def to_dict(self) -> dict:
        return {
            'log_path': str(self.log_path),
            'test_log_path': str(self.test_log_path),
            'log_ids': self.log_ids.to_dict(),
            'model_path': str(self.model_path),
            'exec_mode': str(self.exec_mode),
            'repetitions': self.repetitions,
            'evaluation_metrics': [str(metric) for metric in self.evaluation_metrics],
            'clean_intermediate_files': self.clean_intermediate_files,
            'extraneous_activity_delays': self.extraneous_activity_delays,
        }


@dataclass
class PreprocessingSettings:
    multitasking: bool

    @staticmethod
    def default() -> 'PreprocessingSettings':
        return PreprocessingSettings(
            multitasking=False
        )

    @staticmethod
    def from_dict(config: dict) -> 'PreprocessingSettings':
        return PreprocessingSettings(
            multitasking=config.get('multitasking', False),
        )

    def to_dict(self) -> dict:
        return {
            'multitasking': self.multitasking
        }


@dataclass
class StructureSettings:
    """
    Structure discovery settings.
    """

    optimization_metric: Metric = Metric.DL
    max_evaluations: Optional[int] = None
    mining_algorithm: Optional[StructureMiningAlgorithm] = None
    concurrency: Optional[Union[float, List[float]]] = None
    epsilon: Optional[Union[float, List[float]]] = None  # parallelism threshold (epsilon)
    eta: Optional[Union[float, List[float]]] = None  # percentile for frequency threshold (eta)
    gateway_probabilities: Optional[
        Union[GatewayProbabilitiesDiscoveryMethod, List[GatewayProbabilitiesDiscoveryMethod]]] = None
    replace_or_joins: Optional[Union[bool, List[bool]]] = None  # should replace non-trivial OR joins
    prioritize_parallelism: Optional[Union[bool, List[bool]]] = None  # should prioritize parallelism on loops

    @staticmethod
    def default() -> 'StructureSettings':
        return StructureSettings(
            optimization_metric=Metric.DL,
            max_evaluations=1,
            mining_algorithm=StructureMiningAlgorithm.SPLIT_MINER_2,
            concurrency=None,
            epsilon=[0.0, 1.0],
            eta=[0.0, 1.0],
            gateway_probabilities=GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
            replace_or_joins=False,
            prioritize_parallelism=False,
        )

    @staticmethod
    def from_dict(config: dict) -> 'StructureSettings':
        mining_algorithm = StructureMiningAlgorithm.from_str(config['mining_algorithm'])

        gateway_probabilities = [
            GatewayProbabilitiesDiscoveryMethod.from_str(g)
            for g in config['gateway_probabilities']
        ]

        optimization_metric = config.get('optimization_metric')
        if optimization_metric is not None:
            optimization_metric = Metric.from_str(optimization_metric)
        else:
            optimization_metric = Metric.DL

        return StructureSettings(
            optimization_metric=optimization_metric,
            max_evaluations=config['max_evaluations'],
            mining_algorithm=mining_algorithm,
            concurrency=config['concurrency'],
            epsilon=config['epsilon'],
            eta=config['eta'],
            gateway_probabilities=gateway_probabilities,
            replace_or_joins=config['replace_or_joins'],
            prioritize_parallelism=config['prioritize_parallelism'],
        )

    def to_dict(self) -> dict:
        return {
            'optimization_metric': str(self.optimization_metric),
            'max_evaluations': self.max_evaluations,
            'mining_algorithm': str(self.mining_algorithm),
            'concurrency': self.concurrency,
            'epsilon': self.epsilon,
            'eta': self.eta,
            'gateway_probabilities': [str(g) for g in self.gateway_probabilities],
            'replace_or_joins': self.replace_or_joins,
            'prioritize_parallelism': self.prioritize_parallelism,
        }


@dataclass
class CalendarSettings:
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


@dataclass
class CalendarsSettings:
    optimization_metric: Metric
    max_evaluations: int
    case_arrival: CalendarSettings
    resource_profiles: CalendarSettings

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


@dataclass
class Configuration:
    """
    Simod configuration containing all the settings for structure and calendars optimizations.
    """

    common: CommonSettings
    preprocessing: PreprocessingSettings
    structure: StructureSettings
    calendars: CalendarsSettings

    @staticmethod
    def default() -> 'Configuration':
        """
        Default configuration for Simod. Used mostly for testing purposes. Most of those settings should be discovered
        by Simod automatically.
        """

        return Configuration(
            common=CommonSettings.default(),
            preprocessing=PreprocessingSettings.default(),
            structure=StructureSettings.default(),
            calendars=CalendarsSettings.default()
        )

    @staticmethod
    def from_yaml(config: dict) -> 'Configuration':
        assert config['version'] == 2, 'Configuration version must be 2'

        common_settings = CommonSettings.from_dict(config['common'])
        preprocessing_settings = PreprocessingSettings.from_dict(config['preprocessing'])
        structure_settings = StructureSettings.from_dict(config['structure'])
        calendars_settings = CalendarsSettings.from_dict(config['calendars'])

        # If the model is provided, we don't execute SplitMiner. Then, ignore the mining_algorithm setting
        if common_settings.model_path is not None:
            print_notice(f'Ignoring structure settings because the model is provided')
            structure_settings.mining_algorithm = None
            structure_settings.epsilon = None
            structure_settings.eta = None
            structure_settings.prioritize_parallelism = None
            structure_settings.or_prior = None
            structure_settings.replace_or_joins = None
            structure_settings.concurrency = None

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

    def to_dict(self) -> dict:
        return {
            'version': 2,
            'common': self.common.to_dict(),
            'preprocessing': self.preprocessing.to_dict(),
            'structure': self.structure.to_dict(),
            'calendars': self.calendars.to_dict(),
        }

    def to_yaml(self, output_dir: Path):
        """
        Saves the configuration to a YAML file in the provided output directory.
        :param output_dir: Output directory.
        :return: None.
        """
        data = yaml.dump(self.to_dict(), sort_keys=False)
        output_path = output_dir / 'configuration.yaml'
        with output_path.open('w') as f:
            f.write(data)

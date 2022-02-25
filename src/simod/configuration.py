import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Union

import yaml

from . import support_utils as sup
from .support_utils import get_project_dir

QBP_NAMESPACE_URI = 'http://www.qbp-simulator.com/Schema201212'
BPMN_NAMESPACE_URI = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
PROJECT_DIR = get_project_dir()


class TraceAlignmentAlgorithm(Enum):
    REPLACEMENT = auto()
    REPAIR = auto()
    REMOVAL = auto()

    @classmethod
    def from_str(cls, value: str) -> 'TraceAlignmentAlgorithm':
        if value.lower() == 'replacement':
            return cls.REPLACEMENT
        elif value.lower() == 'repair':
            return cls.REPAIR
        elif value.lower() == 'removal':
            return cls.REMOVAL
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == TraceAlignmentAlgorithm.REPLACEMENT:
            return 'replacement'
        elif self == TraceAlignmentAlgorithm.REPAIR:
            return 'repair'
        elif self == TraceAlignmentAlgorithm.REMOVAL:
            return 'removal'
        return f'Unknown TraceAlignmentAlgorithm {str(self)}'


class MiningAlgorithm(Enum):
    SM1 = auto()
    SM2 = auto()
    SM3 = auto()

    @classmethod
    def from_str(cls, value: str) -> 'MiningAlgorithm':
        if value.lower() == 'sm1':
            return cls.SM1
        elif value.lower() == 'sm2':
            return cls.SM2
        elif value.lower() == 'sm3':
            return cls.SM3
        else:
            raise ValueError(f'Unknown value {value}')


class AndPriorORemove(Enum):
    TRUE = auto()
    FALSE = auto()

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> Union['AndPriorORemove', List['AndPriorORemove']]:
        if isinstance(value, str):
            return AndPriorORemove._from_str(value)
        elif isinstance(value, list):
            return [AndPriorORemove._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'AndPriorORemove':
        if value:
            return cls.TRUE
        elif not value:
            return cls.FALSE
        else:
            raise ValueError(f'Unknown value {value}')

    @classmethod
    def default(cls) -> List['AndPriorORemove']:
        return [AndPriorORemove.FALSE]

    @staticmethod
    def to_str(value: Union['AndPriorORemove', List['AndPriorORemove']]) -> Union[str, List[str]]:
        if isinstance(value, AndPriorORemove):
            return str(value)
        elif isinstance(value, list):
            return [str(item) for item in value]
        else:
            raise ValueError(f'Unknown value type {type(value)}')

    def __str__(self):
        if self == AndPriorORemove.TRUE:
            return 'true'
        elif self == AndPriorORemove.FALSE:
            return 'false'
        return f'Unknown AndPriorORemove {str(self)}'


class GateManagement(Enum):
    DISCOVERY = auto()
    EQUIPROBABLE = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[GateManagement, List[GateManagement]]':
        if isinstance(value, str):
            return GateManagement._from_str(value)
        elif isinstance(value, list):
            return [GateManagement._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'GateManagement':
        if value.lower() == 'discovery':
            return cls.DISCOVERY
        elif value.lower() == 'equiprobable':
            return cls.EQUIPROBABLE
        elif value.lower() == 'random':
            return cls.RANDOM
        else:
            raise ValueError(f'Unknown value {value}')

    def __str__(self):
        if self == GateManagement.DISCOVERY:
            return 'discovery'
        elif self == GateManagement.EQUIPROBABLE:
            return 'equiprobable'
        elif self == GateManagement.RANDOM:
            return 'random'
        return f'Unknown GateManagement {str(self)}'


class CalculationMethod(Enum):
    DEFAULT = auto()
    DISCOVERED = auto()
    POOL = auto()

    @classmethod
    def from_str(cls, value: str) -> 'CalculationMethod':
        if value.lower() == 'default':
            return cls.DEFAULT
        elif value.lower() == 'discovered':
            return cls.DISCOVERED
        elif value.lower() == 'pool':
            return cls.POOL
        else:
            raise ValueError(f'Unknown value {value}')


class DataType(Enum):
    DT247 = auto()
    LV917 = auto()

    @classmethod
    def from_str(cls, value: Union[str, List[str]]) -> 'Union[DataType, List[DataType]]':
        if isinstance(value, str):
            return DataType._from_str(value)
        elif isinstance(value, list):
            return [DataType._from_str(v) for v in value]

    @classmethod
    def _from_str(cls, value: str) -> 'DataType':
        if value == '247' or value.lower() == 'dt247':
            return cls.DT247
        elif value == '917' or value.lower() == 'lv917':
            return cls.LV917
        else:
            raise ValueError(f'Unknown value {value}')


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


class SimulatorKind(Enum):
    BIMP = auto()
    CUSTOM = auto()

    @classmethod
    def from_str(cls, value: str) -> 'SimulatorKind':
        value = value.lower()
        if value in ('bimp', 'qbp'):
            return cls.BIMP
        elif value == 'custom':
            return cls.CUSTOM
        else:
            raise ValueError(f'Unknown value {value}')


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


@dataclass
class ReadOptions:
    column_names: Dict[str, str]
    timeformat: str = '%Y-%m-%dT%H:%M:%S.%f'
    one_timestamp: bool = False  # TODO: must be an obsolete attribute because Simod doesn't work with one_timestamp logs
    filter_d_attrib: bool = True

    @staticmethod
    def column_names_default() -> Dict[str, str]:
        return {'Case ID': 'caseid', 'Activity': 'task', 'lifecycle:transition': 'event_type', 'Resource': 'user'}


@dataclass
class Configuration:
    # General
    project_name: Optional[str] = None
    log_path: Optional[Path] = None
    model_path: Optional[Path] = None
    config_path: Optional[Path] = None
    output: Path = PROJECT_DIR / 'outputs' / sup.folder_id()
    sm1_path: Path = PROJECT_DIR / 'external_tools/splitminer2/sm2.jar'
    sm2_path: Path = PROJECT_DIR / 'external_tools/splitminer2/sm2.jar'
    sm3_path: Path = PROJECT_DIR / 'external_tools/splitminer3/bpmtk.jar'
    bimp_path: Path = PROJECT_DIR / 'external_tools/bimp/qbp-simulator-engine.jar'
    align_path: Path = PROJECT_DIR / 'external_tools/proconformance/ProConformance2.jar'
    calender_path: Path = PROJECT_DIR / 'external_tools/calenderimp/CalenderImp.jar'
    aligninfo: Path = output / 'CaseTypeAlignmentResults.csv'
    aligntype: Path = output / 'AlignmentStatistics.csv'
    read_options: ReadOptions = ReadOptions(column_names=ReadOptions.column_names_default())
    mining_alg: MiningAlgorithm = MiningAlgorithm.SM3
    repetitions: int = 1
    simulator: SimulatorKind = SimulatorKind.BIMP
    simulation_cases: int = 0
    simulation: bool = True  # TODO: is this condition checked anywhere?
    sim_metric: Metric = Metric.TSD
    add_metrics: List[Metric] = field(
        default_factory=lambda: [Metric.DAY_HOUR_EMD, Metric.LOG_MAE, Metric.DL, Metric.MAE])
    concurrency: Union[float, List[float]] = 0.0  # array
    arr_cal_met: CalculationMethod = CalculationMethod.DISCOVERED
    arr_confidence: Optional[Union[float, List[float]]] = None
    arr_support: Optional[Union[float, List[float]]] = None
    epsilon: Optional[Union[float, List[float]]] = None
    eta: Optional[Union[float, List[float]]] = None
    and_prior: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])
    or_rep: List[AndPriorORemove] = field(default_factory=lambda: [AndPriorORemove.FALSE])
    gate_management: Optional[Union[GateManagement, List[GateManagement]]] = None
    res_confidence: Optional[float] = None
    res_support: Optional[float] = None
    res_cal_met: Optional[CalculationMethod] = None
    res_dtype: Optional[Union[DataType, List[DataType]]] = None
    arr_dtype: Optional[Union[DataType, List[DataType]]] = None
    rp_similarity: Optional[Union[float, List[float]]] = None
    pdef_method: Optional[PDFMethod] = None
    multitasking: Optional[bool] = False

    # Optimizer specific
    exec_mode: ExecutionMode = ExecutionMode.SINGLE
    max_eval_s: Optional[int] = None
    max_eval_t: Optional[int] = None
    res_sup_dis: Optional[List[float]] = None
    res_con_dis: Optional[List[float]] = None

    def fill_in_derived_fields(self):
        if self.log_path:
            self.project_name, _ = os.path.splitext(os.path.basename(self.log_path))


def config_data_from_file(config_path: Path) -> dict:
    with config_path.open('r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    if config_data is None:
        raise Exception('Config is empty')
    config_data = config_data_from_yaml(config_data)
    return config_data


def config_data_from_yaml(config_data: dict) -> dict:
    config_data = config_data_with_datastructures(config_data)

    structure_optimizer = config_data.get('structure_optimizer')
    if structure_optimizer:
        structure_optimizer = config_data_with_datastructures(structure_optimizer)
        # the rest of the software uses 'strc' key
        config_data.pop('structure_optimizer')
        config_data['strc'] = structure_optimizer

    time_optimizer = config_data.get('time_optimizer')
    if time_optimizer:
        time_optimizer = config_data_with_datastructures(time_optimizer)
        # the rest of the software uses 'tm' key
        config_data.pop('time_optimizer')
        config_data['tm'] = time_optimizer

    return config_data


def config_data_with_datastructures(data: dict) -> dict:
    global PROJECT_DIR
    data = data.copy()

    model_path = data.get('model_path')
    if model_path:
        data['model_path'] = Path(model_path)

    log_path = data.get('log_path')
    if log_path:
        data['log_path'] = Path(log_path)

    input = data.get('input')
    if input:
        data['input'] = Path(input)

    mining_alg = data.get('mining_alg')
    if mining_alg:
        data['mining_alg'] = MiningAlgorithm.from_str(mining_alg)

    and_prior = data.get('and_prior')
    if and_prior and (isinstance(and_prior, str) or isinstance(and_prior, list)):
        data['and_prior'] = AndPriorORemove.from_str(and_prior)

    or_rep = data.get('or_rep')
    if or_rep:
        data['or_rep'] = AndPriorORemove.from_str(or_rep)

    gate_management = data.get('gate_management')
    if gate_management and (isinstance(gate_management, str) or isinstance(gate_management, list)):
        data['gate_management'] = GateManagement.from_str(gate_management)

    res_cal_met = data.get('res_cal_met')
    if res_cal_met:
        data['res_cal_met'] = CalculationMethod.from_str(res_cal_met)

    res_dtype = data.get('res_dtype')
    if res_dtype:
        data['res_dtype'] = DataType.from_str(res_dtype)

    arr_dtype = data.get('arr_dtype')
    if arr_dtype and (isinstance(arr_dtype, str) or isinstance(arr_dtype, list)):
        data['arr_dtype'] = DataType.from_str(arr_dtype)

    pdef_method = data.get('pdef_method')
    if pdef_method:
        data['pdef_method'] = PDFMethod.from_str(pdef_method)

    exec_mode = data.get('exec_mode')
    if exec_mode:
        data['exec_mode'] = ExecutionMode.from_str(exec_mode)

    sim_metric = data.get('sim_metric')
    if sim_metric:
        data['sim_metric'] = Metric.from_str(sim_metric)

    add_metrics = data.get('add_metrics')
    if add_metrics:
        data['add_metrics'] = Metric.from_str(add_metrics)

    simulator = data.get('simulator')
    if simulator:
        data['simulator'] = SimulatorKind.from_str(simulator)

    return data

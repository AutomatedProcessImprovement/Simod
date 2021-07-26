import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict

import utils.support as sup


class AlgorithmManagement(Enum):
    REPAIR = auto()
    REPLACEMENT = auto()
    REMOVAL = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == 'repair':
            return cls.REPAIR
        elif value == 'removal':
            return cls.REMOVAL
        elif value == 'replacement':
            return cls.REPLACEMENT
        else:
            raise ValueError(f'Unknown value {value}')


class MiningAlgorithm(Enum):
    SM1 = auto()
    SM2 = auto()
    SM3 = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == 'sm1':
            return cls.SM1
        elif value == 'sm2':
            return cls.SM2
        elif value == 'sm3':
            return cls.SM3
        else:
            raise ValueError(f'Unknown value {value}')


class GateManagement(Enum):
    DISCOVERY = auto()
    EQUIPROBABLE = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == 'discovered':
            return cls.DISCOVERY
        elif value == 'equiprobable':
            return cls.EQUIPROBABLE
        elif value == 'random':
            return cls.RANDOM
        else:
            raise ValueError(f'Unknown value {value}')


class CalculationMethod(Enum):
    DEFAULT = auto()
    DISCOVERED = auto()
    POOL = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == 'default':
            return cls.DEFAULT
        elif value == 'discovered':
            return cls.DISCOVERED
        elif value == 'pool':
            return cls.POOL
        else:
            raise ValueError(f'Unknown value {value}')


class DataType(Enum):
    DT247 = auto()
    LV917 = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == '247' or value == 'dt247':
            return cls.DT247
        elif value == 'LV917':
            return cls.LV917
        else:
            raise ValueError(f'Unknown value {value}')


class PDFMethod(Enum):
    AUTOMATIC = auto()
    SEMIAUTOMATIC = auto()
    MANUAL = auto()
    DEFAULT = auto()

    @classmethod
    def from_str(cls, value: str):
        if value == 'automatic':
            return cls.AUTOMATIC
        elif value == 'semiautomatic':
            return cls.SEMIAUTOMATIC
        elif value == 'manual':
            return cls.MANUAL
        elif value == 'default':
            return cls.DEFAULT
        else:
            raise ValueError(f'Unknown value {value}')


class SimulatorKind(Enum):
    BIMP = auto()


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


class ExecutionMode(Enum):
    SINGLE = auto()
    OPTIMIZER = auto()


@dataclass
class ReadOptions:
    column_names: Dict[str, str]
    timeformat: str = '%Y-%m-%dT%H:%M:%S.%f'
    one_timestamp: bool = False
    filter_d_attrib: bool = True

    @staticmethod
    def column_names_default() -> Dict[str, str]:
        return {'Case ID': 'caseid', 'Activity': 'task', 'lifecycle:transition': 'event_type', 'Resource': 'user'}


@dataclass
class Configuration:
    # General
    project_name: str = None
    log_path: Path = None
    model_path: Path = None
    input: Path = None
    file: Path = None
    alg_manag: AlgorithmManagement or List[AlgorithmManagement] = AlgorithmManagement.REPAIR  # [AlgorithmManagement]
    output: Path = os.path.join('outputs', sup.folder_id())
    sm1_path: Path = os.path.join('external_tools', 'splitminer2', 'sm2.jar')
    sm2_path: Path = os.path.join('external_tools', 'splitminer2', 'sm2.jar')
    sm3_path: Path = os.path.join('external_tools', 'splitminer3', 'bpmtk.jar')
    bimp_path: Path = os.path.join('external_tools', 'bimp', 'qbp-simulator-engine.jar')
    align_path: Path = os.path.join('external_tools', 'proconformance', 'ProConformance2.jar')
    calender_path: Path = os.path.join('external_tools', 'calenderimp', 'CalenderImp.jar')
    aligninfo: Path = os.path.join(output, 'CaseTypeAlignmentResults.csv')
    aligntype: Path = os.path.join(output, 'AlignmentStatistics.csv')
    read_options: ReadOptions = ReadOptions(column_names=ReadOptions.column_names_default())
    simulator: SimulatorKind = SimulatorKind.BIMP
    mining_alg: MiningAlgorithm = MiningAlgorithm.SM3
    repetitions: int = 1
    simulation: bool = True
    sim_metric: Metric = Metric.TSD
    add_metrics: List[Metric] = field(
        default_factory=lambda: [Metric.DAY_HOUR_EMD, Metric.LOG_MAE, Metric.DL, Metric.MAE])
    concurrency: float or List[float] = 0.0  # array
    arr_cal_met: CalculationMethod = CalculationMethod.DISCOVERED
    arr_confidence: float or List[float] = None
    arr_support: float or List[float] = None
    epsilon: float or List[float] = None
    eta: float or List[float] = None
    gate_management: GateManagement or List[GateManagement] = None
    res_confidence: float = None
    res_support: float = None
    res_cal_met: CalculationMethod = None
    res_dtype: DataType or List[DataType] = None
    arr_dtype: DataType or List[DataType] = None
    rp_similarity: float or List[float] = None
    pdef_method: PDFMethod = None

    # Optimizer specific
    exec_mode: ExecutionMode = ExecutionMode.SINGLE
    max_eval_s: int = None
    max_eval_t: int = None
    res_sup_dis: List[float] = None
    res_con_dis: List[float] = None
    new_replayer: bool = False

    def fill_in_derived_fields(self):
        if self.log_path:
            self.input = os.path.dirname(self.log_path)
            self.file = os.path.basename(self.log_path)
            self.project_name, _ = os.path.splitext(os.path.basename(self.log_path))

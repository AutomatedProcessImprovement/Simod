from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List, Tuple

import yaml

from simod.configuration import PDFMethod, GateManagement, DataType, ProjectSettings


@dataclass
class CalendarOptimizationSettings:
    """Settings for resources' and arrival calendars optimizer."""
    max_evaluations: int = 1
    simulation_repetitions: int = 1
    pdef_method: Optional[PDFMethod] = PDFMethod.AUTOMATIC
    gateway_probabilities: Optional[Union[GateManagement, List[GateManagement]]] = GateManagement.DISCOVERY

    rp_similarity: Union[float, list[float], None] = None
    res_sup_dis: Optional[list[float]] = None
    res_con_dis: Optional[list[float]] = None
    res_dtype: Union[DataType, list[DataType], None] = None
    arr_support: Union[float, list[float], None] = None
    arr_confidence: Union[float, list[float], None] = None
    arr_dtype: Union[DataType, list[DataType], None] = None

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> 'CalendarOptimizationSettings':
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        v = settings.get('time_optimizer', None)
        if v is not None:
            settings = v

        max_evaluations = settings.get('max_evaluations', None)
        if max_evaluations is None:
            max_evaluations = settings.get('max_eval_t', 1)  # legacy support

        simulation_repetitions = settings.get('simulation_repetitions', 1)

        pdef_method = settings.get('pdef_method', PDFMethod.AUTOMATIC)

        gateway_probabilities = settings.get('gateway_probabilities', None)
        if gateway_probabilities is None:
            gateway_probabilities = settings.get('gate_management', None)  # legacy key support
        if gateway_probabilities is not None:
            if isinstance(gateway_probabilities, list):
                gateway_probabilities = [GateManagement.from_str(g) for g in gateway_probabilities]
            elif isinstance(gateway_probabilities, str):
                gateway_probabilities = GateManagement.from_str(gateway_probabilities)
            else:
                raise ValueError('Gateway probabilities must be a list or a string.')

        rp_similarity = settings.get('rp_similarity', None)
        res_sup_dis = settings.get('res_sup_dis', None)
        res_con_dis = settings.get('res_con_dis', None)
        res_dtype = settings.get('res_dtype', None)
        if res_dtype is not None:
            res_dtype = DataType.from_str(res_dtype)  # TODO: this should change with new calendars
        arr_support = settings.get('arr_support', None)
        arr_confidence = settings.get('arr_confidence', None)
        arr_dtype = settings.get('arr_dtype', None)  # TODO: this should change with new calendars
        if arr_dtype is not None:
            arr_dtype = DataType.from_str(arr_dtype)

        return CalendarOptimizationSettings(
            max_evaluations=max_evaluations,
            simulation_repetitions=simulation_repetitions,
            pdef_method=pdef_method,
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_sup_dis=res_sup_dis,
            res_con_dis=res_con_dis,
            res_dtype=res_dtype,
            arr_support=arr_support,
            arr_confidence=arr_confidence,
            arr_dtype=arr_dtype)


@dataclass
class ResourceOptimizationSettings:
    # in case of "discovered"
    res_confidence: Optional[float] = None
    res_support: Optional[float] = None

    # in case of "default"
    res_dtype: Optional[DataType] = None

    def __post_init__(self):
        assert (self.res_confidence is not None and self.res_support is not None) or (self.res_dtype is not None), \
            'Either resource confidence and support or calendar type should be specified'


@dataclass
class ArrivalOptimizationSettings:
    # in case of "discovered"
    arr_confidence: Optional[float] = None
    arr_support: Optional[float] = None

    # in case of "default"
    arr_dtype: Optional[DataType] = None

    def __post_init__(self):
        assert (self.arr_confidence is not None and self.arr_support is not None) or (self.arr_dtype is not None), \
            'Either arrival confidence and support or calendar type should be specified'


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
class PipelineSettings(ProjectSettings):
    """Settings for the calendars optimizer pipeline."""
    gateway_probabilities: Optional[GateManagement]
    rp_similarity: float
    res_cal_met: Tuple[CalendarOptimizationType, ResourceOptimizationSettings]
    arr_cal_met: Tuple[CalendarOptimizationType, ArrivalOptimizationSettings]

    @staticmethod
    def from_dict(data: dict) -> 'PipelineSettings':
        project_settings = ProjectSettings.from_dict(data)

        rp_similarity = data.get('rp_similarity', None)
        assert rp_similarity is not None, 'rp_similarity is not specified'

        res_cal_met = data.get('res_cal_met', None)
        assert res_cal_met is not None, 'res_cal_met is not specified'

        arr_cal_met = data.get('arr_cal_met', None)
        assert arr_cal_met is not None, 'arr_cal_met is not specified'

        resource_optimization_type = CalendarOptimizationType.from_str(res_cal_met[0])
        if resource_optimization_type == CalendarOptimizationType.DISCOVERED:
            res_confidence = res_cal_met[1].get('res_confidence', None)
            assert res_confidence is not None, 'res_confidence is not specified'

            res_support = res_cal_met[1].get('res_support', None)
            assert res_support is not None, 'res_support is not specified'

            resource_settings = ResourceOptimizationSettings(res_confidence, res_support)
        elif resource_optimization_type == CalendarOptimizationType.DEFAULT:
            res_dtype = res_cal_met[1].get('res_dtype', None)
            assert res_dtype is not None, 'res_dtype is not specified'

            resource_settings = ResourceOptimizationSettings(res_dtype=res_dtype)
        else:
            raise ValueError(f'Unknown optimization type: {resource_optimization_type}')

        arrival_optimization_type = CalendarOptimizationType.from_str(arr_cal_met[0])
        if arrival_optimization_type == CalendarOptimizationType.DISCOVERED:
            arr_confidence = arr_cal_met[1].get('arr_confidence', None)
            assert arr_confidence is not None, 'arr_confidence is not specified'

            arr_support = arr_cal_met[1].get('arr_support', None)
            assert arr_support is not None, 'arr_support is not specified'

            arrival_settings = ArrivalOptimizationSettings(arr_confidence, arr_support)
        elif arrival_optimization_type == CalendarOptimizationType.DEFAULT:
            arr_dtype = arr_cal_met[1].get('arr_dtype', None)
            assert arr_dtype is not None, 'arr_dtype is not specified'

            arrival_settings = ArrivalOptimizationSettings(arr_dtype=arr_dtype)
        else:
            raise ValueError(f'Unknown optimization type: {arrival_optimization_type}')

        gateway_probabilities = data.get('gateway_probabilities', None)

        return PipelineSettings(
            **project_settings.__dict__,
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_cal_met=(resource_optimization_type, resource_settings),
            arr_cal_met=(arrival_optimization_type, arrival_settings)
        )

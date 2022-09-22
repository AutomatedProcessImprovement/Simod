from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple

import yaml

from simod.configuration import PDFMethod, GateManagement, DataType


@dataclass
class CalendarOptimizationSettings:
    """Settings for resources' and arrival calendars optimizer."""
    base_dir: Optional[Path]

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
    def from_stream(stream: Union[str, bytes], base_dir: Path) -> 'CalendarOptimizationSettings':
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
            base_dir=base_dir,
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
class PipelineSettings:
    """Settings for the calendars optimizer pipeline."""
    # General settings
    output_dir: Path  # each pipeline run creates its own directory
    model_path: Path  # in calendars optimizer, this path doesn't change and just inherits from the project settings

    # Optimization settings
    gateway_probabilities: Optional[GateManagement]
    rp_similarity: float
    res_cal_met: Tuple[CalendarOptimizationType, ResourceOptimizationSettings]
    arr_cal_met: Tuple[CalendarOptimizationType, ArrivalOptimizationSettings]

    @staticmethod
    def from_hyperopt_response(
            data: dict,
            initial_settings: CalendarOptimizationSettings,
            output_dir: Path,
            model_path: Path,
    ) -> 'PipelineSettings':
        gateway_probabilities = data.get('gateway_probabilities', None)
        assert gateway_probabilities is not None, 'Gateway probabilities must be specified'
        gateway_probabilities = initial_settings.gateway_probabilities[gateway_probabilities]

        rp_similarity = data.get('rp_similarity', None)

        resource_calendar_discovery_type = data.get('res_cal_met', None)
        assert resource_calendar_discovery_type is not None, 'Resource calendar optimization method is not specified'
        if resource_calendar_discovery_type == 0:  # 0 is an index of the tuple that was provided to hyperopt in search space
            resource_calendar_discovery_type = CalendarOptimizationType.DISCOVERED
        elif resource_calendar_discovery_type == 1:
            resource_calendar_discovery_type = CalendarOptimizationType.DEFAULT
        else:
            raise ValueError(f'Unknown resource calendar optimization method: {resource_calendar_discovery_type}')

        if resource_calendar_discovery_type == CalendarOptimizationType.DISCOVERED:
            confidence = data.get('res_confidence', None)
            support = data.get('res_support', None)
            resource_settings = ResourceOptimizationSettings(res_confidence=confidence, res_support=support)
        elif resource_calendar_discovery_type == CalendarOptimizationType.DEFAULT:
            dtype = initial_settings.res_dtype[data.get('res_dtype', None)]
            resource_settings = ResourceOptimizationSettings(res_dtype=dtype)
        else:
            raise ValueError(f'Unknown resource calendar optimization method: {resource_calendar_discovery_type}')

        res_cal_met = (resource_calendar_discovery_type, resource_settings)

        arrival_calendar_discovery_type = data.get('arr_cal_met', None)
        assert arrival_calendar_discovery_type is not None, 'Arrival calendar optimization method is not specified'
        if arrival_calendar_discovery_type == 0:  # 0 is an index of the tuple that was provided to hyperopt in search space
            arrival_calendar_discovery_type = CalendarOptimizationType.DISCOVERED
        elif arrival_calendar_discovery_type == 1:
            arrival_calendar_discovery_type = CalendarOptimizationType.DEFAULT
        else:
            raise ValueError(f'Unknown arrival calendar optimization method: {arrival_calendar_discovery_type}')

        if arrival_calendar_discovery_type == CalendarOptimizationType.DISCOVERED:
            confidence = data.get('arr_confidence', None)
            support = data.get('arr_support', None)
            arrival_settings = ArrivalOptimizationSettings(arr_confidence=confidence, arr_support=support)
        elif arrival_calendar_discovery_type == CalendarOptimizationType.DEFAULT:
            dtype = initial_settings.arr_dtype[data.get('arr_dtype', None)]
            arrival_settings = ArrivalOptimizationSettings(arr_dtype=dtype)
        else:
            raise ValueError(f'Unknown arrival calendar optimization method: {arrival_calendar_discovery_type}')

        arr_cal_met = (arrival_calendar_discovery_type, arrival_settings)

        return PipelineSettings(
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_cal_met=res_cal_met,
            arr_cal_met=arr_cal_met,
            output_dir=output_dir,
            model_path=model_path,
        )

    @staticmethod
    def from_dict(data: dict, output_dir: Path, model_path: Path) -> 'PipelineSettings':
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
        assert gateway_probabilities is not None, 'gateway_probabilities is not specified'

        return PipelineSettings(
            gateway_probabilities=gateway_probabilities,
            rp_similarity=rp_similarity,
            res_cal_met=(resource_optimization_type, resource_settings),
            arr_cal_met=(arrival_optimization_type, arrival_settings),
            output_dir=output_dir,
            model_path=model_path,
        )

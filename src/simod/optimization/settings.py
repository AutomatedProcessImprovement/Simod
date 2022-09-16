from dataclasses import dataclass, field
from typing import List

from simod.configuration import ProjectSettings, Metric
from simod.process_calendars.settings import CalendarOptimizationSettings
from simod.process_structure.settings import StructureOptimizationSettings


@dataclass
class OptimizationSettings:
    project_settings: ProjectSettings
    structure_settings: StructureOptimizationSettings
    calendar_settings: CalendarOptimizationSettings

    num_simulations: int = 2
    adjust_for_multitasking: bool = False
    discover_structure: bool = True
    evaluation_metrics: List[Metric] = field(
        default_factory=lambda: [Metric.DAY_HOUR_EMD, Metric.LOG_MAE, Metric.DL, Metric.MAE])

    def __post_init__(self):
        self.structure_settings.project_name = self.project_settings.project_name  # TODO: refactor this

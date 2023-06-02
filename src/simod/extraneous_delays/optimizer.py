from pathlib import Path
from typing import Union, List

from extraneous_activity_delays.config import (
    Configuration as ExtraneousActivityDelaysConfiguration,
    SimulationModel,
)
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from lxml import etree

from simod.event_log.event_log import EventLog
from simod.settings.extraneous_delays_settings import ExtraneousDelaysSettings
from simod.simulation.parameters.BPS_model import BPSModel
from simod.simulation.parameters.extraneous_delays import ExtraneousDelay


class ExtraneousDelayTimersOptimizer:
    def __init__(
        self,
        event_log: EventLog,
        bps_model: BPSModel,
        settings: Union[ExtraneousDelaysSettings, None],
        base_directory: Path,
    ):
        self.event_log = event_log
        self.bps_model = bps_model
        self.settings = settings
        self.base_directory = base_directory

        assert self.bps_model.process_model is not None, "BPMN model is not specified."

    def run(self) -> List[ExtraneousDelay]:
        configuration = ExtraneousActivityDelaysConfiguration(
            log_ids=self.event_log.log_ids,
            process_name=self.event_log.process_name,
            num_iterations=self.settings.num_iterations,
            optimization_metric=self.settings.optimization_metric,
        )

        parser = etree.XMLParser(remove_blank_text=True)
        bpmn_model = etree.parse(self.bps_model.process_model, parser)
        parameters = self.bps_model.to_dict()

        simulation_model = SimulationModel(bpmn_model, parameters)

        enhancer = HyperOptEnhancer(self.event_log.train_partition, simulation_model, configuration)

        timers = [
            ExtraneousDelay(activity_name, distribution) for activity_name, distribution in enhancer.best_timers.items()
        ]

        return timers

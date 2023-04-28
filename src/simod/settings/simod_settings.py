from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel

from .common_settings import CommonSettings
from .control_flow_settings import ControlFlowSettings
from .extraneous_delays_settings import ExtraneousDelaysSettings
from .preprocessing_settings import PreprocessingSettings
from .temporal_settings import CalendarsSettings
from ..cli_formatter import print_notice
from ..utilities import get_project_dir

QBP_NAMESPACE_URI = 'http://www.qbp-simulator.com/Schema201212'
BPMN_NAMESPACE_URI = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
PROJECT_DIR = get_project_dir()


class SimodSettings(BaseModel):
    """
    Simod configuration containing all the settings for structure and calendars optimizations.
    """

    common: CommonSettings
    preprocessing: PreprocessingSettings
    structure: ControlFlowSettings
    calendars: CalendarsSettings
    extraneous_activity_delays: Union[ExtraneousDelaysSettings, None] = None

    @staticmethod
    def default() -> 'SimodSettings':
        """
        Default configuration for Simod. Used mostly for testing purposes. Most of those settings should be discovered
        by Simod automatically.
        """

        return SimodSettings(
            common=CommonSettings.default(),
            preprocessing=PreprocessingSettings.default(),
            structure=ControlFlowSettings(),
            calendars=CalendarsSettings.default(),
            extraneous_activity_delays=ExtraneousDelaysSettings.default()
        )

    @staticmethod
    def from_yaml(config: dict) -> 'SimodSettings':
        assert config['version'] == 2, 'Configuration version must be 2'

        common_settings = CommonSettings.from_dict(config['common'])
        preprocessing_settings = PreprocessingSettings.from_dict(config['preprocessing'])
        structure_settings = ControlFlowSettings.from_dict(config['structure'])
        calendars_settings = CalendarsSettings.from_dict(config['calendars'])
        extraneous_activity_delays_settings = ExtraneousDelaysSettings.from_dict(
            config.get('extraneous_activity_delays'))

        # If the model is provided, we don't execute SplitMiner. Then, ignore the mining_algorithm setting
        if common_settings.model_path is not None:
            print_notice(f'Ignoring structure settings because the model is provided')
            structure_settings.mining_algorithm = None
            structure_settings.epsilon = None
            structure_settings.eta = None
            structure_settings.prioritize_parallelism = None
            structure_settings.replace_or_joins = None
            structure_settings.concurrency = None

        return SimodSettings(
            common=common_settings,
            preprocessing=preprocessing_settings,
            structure=structure_settings,
            calendars=calendars_settings,
            extraneous_activity_delays=extraneous_activity_delays_settings
        )

    @staticmethod
    def from_stream(stream) -> 'SimodSettings':
        import yaml
        config = yaml.safe_load(stream)
        return SimodSettings.from_yaml(config)

    @staticmethod
    def from_path(file_path: Path) -> 'SimodSettings':
        with file_path.open() as f:
            return SimodSettings.from_stream(f)

    def to_dict(self) -> dict:
        return {
            'version': 2,
            'common': self.common.to_dict(),
            'preprocessing': self.preprocessing.to_dict(),
            'structure': self.structure.to_dict(),
            'calendars': self.calendars.to_dict(),
        }

    def to_yaml(self, output_dir: Path) -> Path:
        """
        Saves the configuration to a YAML file in the provided output directory.
        :param output_dir: Output directory.
        :return: None.
        """
        data = yaml.dump(self.to_dict(), sort_keys=False)
        output_path = output_dir / 'configuration.yaml'
        with output_path.open('w') as f:
            f.write(data)
        return output_path

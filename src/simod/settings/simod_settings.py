import copy
from pathlib import Path
from typing import Union, Optional

import yaml
from pydantic import BaseModel

from .common_settings import CommonSettings
from .control_flow_settings import ControlFlowSettings
from .extraneous_delays_settings import ExtraneousDelaysSettings
from .preprocessing_settings import PreprocessingSettings
from .resource_model_settings import ResourceModelSettings
from ..cli_formatter import print_notice

QBP_NAMESPACE_URI = "http://www.qbp-simulator.com/Schema201212"
BPMN_NAMESPACE_URI = "http://www.omg.org/spec/BPMN/20100524/MODEL"


class SimodSettings(BaseModel):
    """
    Simod configuration v4 with the settings for all the stages and optimizations.
    If configuration is provided in v2, is transformed to v4.
    """

    common: CommonSettings = CommonSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    control_flow: ControlFlowSettings = ControlFlowSettings()
    resource_model: ResourceModelSettings = ResourceModelSettings()
    extraneous_activity_delays: Union[ExtraneousDelaysSettings, None] = None
    version: int = 4

    @staticmethod
    def default() -> "SimodSettings":
        """
        Default configuration for Simod. Used mostly for testing purposes. Most of those settings should be discovered
        by Simod automatically.
        """

        return SimodSettings(
            common=CommonSettings(),
            preprocessing=PreprocessingSettings(),
            control_flow=ControlFlowSettings(),
            resource_model=ResourceModelSettings(),
            extraneous_activity_delays=ExtraneousDelaysSettings(),
        )

    @staticmethod
    def one_shot() -> "SimodSettings":
        return SimodSettings(
            common=CommonSettings(),
            preprocessing=PreprocessingSettings(),
            control_flow=ControlFlowSettings.one_shot(),
            resource_model=ResourceModelSettings.one_shot(),
            extraneous_activity_delays=ExtraneousDelaysSettings(),
        )

    @staticmethod
    def from_yaml(config: dict, config_dir: Optional[Path] = None) -> "SimodSettings":
        assert config["version"] in [2, 4], "Configuration version must be 2 or 4"

        # Transform from previous version to the latest if needed
        if config["version"] == 2:
            config = _parse_legacy_config(config)

        # Get each of the settings components if present, default otherwise
        if "common" in config:
            common_settings = CommonSettings.from_dict(config["common"], config_dir=config_dir)
        else:
            print_notice("No 'common' settings provided, running Simod with default values.")
            common_settings = CommonSettings()
        if "preprocessing" in config:
            preprocessing_settings = PreprocessingSettings.from_dict(config["preprocessing"])
        else:
            preprocessing_settings = PreprocessingSettings()
        if "control_flow" in config:
            control_flow_settings = ControlFlowSettings.from_dict(config["control_flow"])
        else:
            print_notice("No 'control_flow' settings provided, running Simod with default values.")
            control_flow_settings = ControlFlowSettings()
        if "resource_model" in config:
            resource_model_settings = ResourceModelSettings.from_dict(config["resource_model"])
        else:
            print_notice("No 'resource_model' settings provided, running Simod with default values.")
            resource_model_settings = ResourceModelSettings()
        if "extraneous_activity_delays" in config:
            extraneous_delays_settings = ExtraneousDelaysSettings.from_dict(config["extraneous_activity_delays"])
        else:
            extraneous_delays_settings = None

        # If the model is provided, we don't execute SplitMiner, ignore mining_algorithm settings
        if common_settings.process_model_path is not None:
            print_notice("Ignoring process model discovery settings (the model is provided)")
            control_flow_settings.mining_algorithm = None
            control_flow_settings.epsilon = None
            control_flow_settings.eta = None
            control_flow_settings.prioritize_parallelism = None
            control_flow_settings.replace_or_joins = None

        return SimodSettings(
            version=config["version"],
            common=common_settings,
            preprocessing=preprocessing_settings,
            control_flow=control_flow_settings,
            resource_model=resource_model_settings,
            extraneous_activity_delays=extraneous_delays_settings,
        )

    @staticmethod
    def from_path(file_path: Path) -> "SimodSettings":
        with file_path.open() as f:
            config = yaml.safe_load(f)
            return SimodSettings.from_yaml(config, config_dir=file_path.parent)

    def to_dict(self) -> dict:
        dictionary = {
            "version": self.version,
            "common": self.common.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "control_flow": self.control_flow.to_dict(),
            "resource_model": self.resource_model.to_dict(),
        }
        if self.extraneous_activity_delays is not None:
            dictionary["extraneous_activity_delays"] = self.extraneous_activity_delays.to_dict()
        return dictionary

    def to_yaml(self, output_dir: Path) -> Path:
        """
        Saves the configuration to a YAML file in the provided output directory.
        :param output_dir: Output directory.
        :return: None.
        """
        data = yaml.dump(self.to_dict(), sort_keys=False)
        output_path = output_dir / "configuration.yaml"
        with output_path.open("w") as f:
            f.write(data)
        return output_path


def _parse_legacy_config(config: dict) -> dict:
    parsed_config = copy.deepcopy(config)
    if config["version"] == 2:
        # Transform dictionary from version 2 to 4
        parsed_config["version"] = 4
        # Common elements
        if "log_path" in parsed_config["common"]:
            parsed_config["common"]["train_log_path"] = parsed_config["common"]["log_path"]
            del parsed_config["common"]["log_path"]
            if "repetitions" in parsed_config["common"]:
                parsed_config["common"]["num_final_evaluations"] = parsed_config["common"]["repetitions"]
                del parsed_config["common"]["repetitions"]
        # Control-flow model
        if "structure" in parsed_config:
            parsed_config["control_flow"] = parsed_config["structure"]
            del parsed_config["structure"]
        if "control_flow" in parsed_config:
            if "max_evaluations" in parsed_config["control_flow"]:
                parsed_config["control_flow"]["num_iterations"] = parsed_config["control_flow"]["max_evaluations"]
                del parsed_config["control_flow"]["max_evaluations"]
            if "or_rep" in parsed_config["control_flow"]:
                parsed_config["control_flow"]["replace_or_joins"] = parsed_config["control_flow"]["or_rep"]
                del parsed_config["control_flow"]["or_rep"]
            if "and_prior" in parsed_config["control_flow"]:
                parsed_config["control_flow"]["prioritize_parallelism"] = parsed_config["control_flow"]["and_prior"]
                del parsed_config["control_flow"]["and_prior"]
        # Resource model
        if "calendars" in parsed_config:
            parsed_config["resource_model"] = parsed_config["calendars"]
            del parsed_config["calendars"]
        if "resource_model" in parsed_config:
            if "max_evaluations" in parsed_config["resource_model"]:
                parsed_config["resource_model"]["num_iterations"] = parsed_config["resource_model"]["max_evaluations"]
                del parsed_config["resource_model"]["max_evaluations"]
            if "case_arrival" in parsed_config["resource_model"]:
                del parsed_config["resource_model"]["case_arrival"]
    # Return parsed configuration
    return parsed_config

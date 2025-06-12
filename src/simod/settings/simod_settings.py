import copy
from pathlib import Path
from typing import Union, Optional

import yaml
from pydantic import BaseModel

from .case_arrival_settings import CaseArrivalSettings
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
    SIMOD configuration v5.2 with the settings for all the stages and optimizations.
    If configuration is provided in v2, v4, or v5 it is automatically translated to v5.2.

    Attributes
    ----------
        common : :class:`~simod.settings.common_settings.CommonSettings`
            General configuration parameters of SIMOD and parameters common to all pipeline stages.
        preprocessing : :class:`~simod.settings.preprocessing_settings.PreprocessingSettings`
            Configuration parameters for the preprocessing stage of SIMOD.
        control_flow : :class:`~simod.settings.control_flow_settings.ControlFlowSettings`
            Configuration parameters for the control-flow model discovery stage.
        resource_model : :class:`~simod.settings.resource_model_settings.ResourceModelSettings`
            Configuration parameters for the resource model discovery stage.
        extraneous_activity_delays : :class:`~simod.settings.extraneous_delays_settings.ExtraneousDelaysSettings`
            Configuration parameters for the extraneous delays model discovery stage. If not provided, the extraneous
            delays are not discovered.
        version : float
            SIMOD version.
    """

    common: CommonSettings = CommonSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    case_arrival: CaseArrivalSettings = CaseArrivalSettings()
    control_flow: ControlFlowSettings = ControlFlowSettings()
    resource_model: ResourceModelSettings = ResourceModelSettings()
    extraneous_activity_delays: Union[ExtraneousDelaysSettings, None] = None
    version: float = 5.2

    @staticmethod
    def default() -> "SimodSettings":
        """
        Default configuration for SIMOD.

        Returns
        -------
        :class:`SimodSettings`
            Instance of the SIMOD configuration with the default values.
        """

        return SimodSettings(
            common=CommonSettings(),
            preprocessing=PreprocessingSettings(),
            case_arrival=CaseArrivalSettings(),
            control_flow=ControlFlowSettings(),
            resource_model=ResourceModelSettings(),
            extraneous_activity_delays=ExtraneousDelaysSettings(),
        )

    @staticmethod
    def one_shot() -> "SimodSettings":
        """
        Configuration for SIMOD one-shot. This mode runs SIMOD without optimizing each BPS model component (i.e.,
        directly discover each BPS model component with default parameters).

        Returns
        -------
        :class:`SimodSettings`
            Instance of the SIMOD configuration for one-shot mode.
        """
        return SimodSettings(
            common=CommonSettings(),
            preprocessing=PreprocessingSettings(),
            case_arrival=CaseArrivalSettings.one_shot(),
            control_flow=ControlFlowSettings.one_shot(),
            resource_model=ResourceModelSettings.one_shot(),
            extraneous_activity_delays=ExtraneousDelaysSettings(),
        )

    @staticmethod
    def from_yaml(config: dict, config_dir: Optional[Path] = None) -> "SimodSettings":
        """
        Instantiates the SIMOD configuration from a dictionary following the expected YAML structure.

        Parameters
        ----------
        config : dict
            Dictionary with the configuration values for each of the SIMOD elements.
        config_dir : :class:`~pathlib.Path`, optional
            If the path to the event log(s) is specified in a relative manner, ``[config_dir]`` is used to complete
            such paths. If ``None``, relative paths are complemented with the current directory.

        Returns
        -------
        :class:`SimodSettings`
            Instance of the SIMOD configuration for the specified dictionary values.
        """
        assert config["version"] in [2, 4, 5, 5.2], "Configuration version must be 2, 4, 5, or 5.2"

        # Transform from previous version to the latest if needed
        if config["version"] == 2:
            config = _parse_legacy_config_2(config)
        elif config["version"] == 4:
            config = _parse_legacy_config_4(config)
        elif config["version"] == 5:
            config = _parse_legacy_config_5(config)

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
        if "case_arrival" in config:
            case_arrival_settings = CaseArrivalSettings.from_dict(config["case_arrival"])
        else:
            print_notice("No 'case_arrival' settings provided, running Simod with default values.")
            case_arrival_settings = CaseArrivalSettings()
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
            case_arrival=case_arrival_settings,
            control_flow=control_flow_settings,
            resource_model=resource_model_settings,
            extraneous_activity_delays=extraneous_delays_settings,
        )

    @staticmethod
    def from_path(file_path: Path) -> "SimodSettings":
        """
        Instantiates the SIMOD configuration from a YAML file.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path`
            Path to the YAML file storing the configuration.

        Returns
        -------
        :class:`SimodSettings`
            Instance of the SIMOD configuration for the specified YAML file.
        """
        with file_path.open() as f:
            config = yaml.safe_load(f)
            return SimodSettings.from_yaml(config, config_dir=file_path.parent)

    def to_dict(self) -> dict:
        """
        Translate the SIMOD configuration stored in this instance into a dictionary.

        Returns
        -------
        dict
            Python dictionary storing this configuration.
        """
        dictionary = {
            "version": self.version,
            "common": self.common.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "case_arrival": self.case_arrival.to_dict(),
            "control_flow": self.control_flow.to_dict(),
            "resource_model": self.resource_model.to_dict(),
        }
        if self.extraneous_activity_delays is not None:
            dictionary["extraneous_activity_delays"] = self.extraneous_activity_delays.to_dict()
        return dictionary

    def to_yaml(self, output_dir: Path) -> Path:
        """
        Saves the configuration to a YAML file in the provided output directory.

        Parameters
        ----------
        output_dir : :class:`~pathlib.Path`
            Path to the output directory where to store the YAML file with the configuration.

        Returns
        -------
        :class:`~pathlib.Path`
            Path to the YAML file with the configuration.
        """
        data = yaml.dump(self.to_dict(), sort_keys=False)
        output_path = output_dir / "configuration.yaml"
        with output_path.open("w") as f:
            f.write(data)
        return output_path


def _parse_legacy_config_2(config: dict) -> dict:
    parsed_config = copy.deepcopy(config)
    if config["version"] == 2:
        # Transform dictionary from version 2 to 5.2
        parsed_config["version"] = 5.2
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


def _parse_legacy_config_4(config: dict) -> dict:
    parsed_config = copy.deepcopy(config)
    if config["version"] == 4:
        # Transform dictionary from version 4 to 5.2
        parsed_config["version"] = 5
        # Common elements
        if "discover_case_attributes" in parsed_config["common"]:
            parsed_config["common"]["discover_data_attributes"] = parsed_config["common"]["discover_case_attributes"]
            del parsed_config["common"]["discover_case_attributes"]
        # Transform from v5 to v5.2
        parsed_config = _parse_legacy_config_5(parsed_config)
    # Return parsed configuration
    return parsed_config


def _parse_legacy_config_5(config: dict) -> dict:
    parsed_config = copy.deepcopy(config)
    if config["version"] == 5:
        # Transform dictionary from version 4 to 5
        parsed_config["version"] = 5.2
        # Common elements
        if "use_observed_arrival_distribution" in parsed_config["common"]:
            parsed_config["case_arrival"] = {
                "use_observed_arrival_distribution": parsed_config["common"]["use_observed_arrival_distribution"]
            }
            del parsed_config["common"]["use_observed_arrival_distribution"]
    # Return parsed configuration
    return parsed_config

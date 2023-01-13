import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from lxml import etree

from extraneous_activity_delays.config import Configuration as ExtraneousActivityDelaysConfiguration, \
    SimulationEngine, SimulationModel, OptimizationMetric
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from simod.event_log.column_mapping import EventLogIDs
from simod.simulation.prosimos import SimulationParameters


def discover_extraneous_delay_timers(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        model_path: Path,
        simulation_parameters: Union[SimulationParameters, dict],
        optimization_metric: OptimizationMetric,
        base_dir: Optional[Path] = None,
        num_iterations: int = 50,
        max_alpha: float = 50,
) -> [SimulationModel, Path, Path]:
    """
    Adds extraneous delay timers to the BPMN model and event distribution parameters to the simulation parameters.
    The results are saved to the corresponding files.

    See details at https://github.com/AutomatedProcessImprovement/extraneous-activity-delays.

    :param event_log: Event log.
    :param log_ids: Event log IDs.
    :param model_path: BPMN model path.
    :param simulation_parameters: Prosimos simulation parameters.
    :param optimization_metric: Optimization metric.
    :param base_dir: Base directory for the new model and simulation parameters files.
    :param num_iterations: Number of iterations for the hyperparameter optimization.
    :param max_alpha: Maximum scale factor to multiply the discovered timers in the hyper-optimization.
    :return: Enhanced simulation model and paths to the BPMN model with extraneous delay timers and simulation parameters.
    """
    if base_dir is None:
        base_dir = model_path.parent

    configuration = ExtraneousActivityDelaysConfiguration(
        log_ids=log_ids,
        process_name=model_path.stem,
        max_alpha=max_alpha,
        num_iterations=num_iterations,
        simulation_engine=SimulationEngine.PROSIMOS,
        optimization_metric=optimization_metric,
    )

    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse(model_path, parser)

    parameters = simulation_parameters \
        if isinstance(simulation_parameters, dict) \
        else simulation_parameters.to_dict()

    simulation_model = SimulationModel(bpmn_model, parameters)

    enhancer = HyperOptEnhancer(event_log, simulation_model, configuration)
    enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()

    output_model_path = base_dir / model_path.with_stem(model_path.stem + '_timers').name
    enhanced_simulation_model.bpmn_document.write(output_model_path, pretty_print=True)

    output_parameters_path = base_dir / 'simulation_parameters_with_timers.json'
    with output_parameters_path.open('w') as f:
        json.dump(enhanced_simulation_model.simulation_parameters, f)

    return enhanced_simulation_model, output_model_path, output_parameters_path

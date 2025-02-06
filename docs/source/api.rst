API Reference
=============

This section provides an overview of the Simod API.

Usage
-----
To use Simod in your Python code, import the main components:

.. code-block:: python

    from pathlib import Path

    from simod.event_log.event_log import EventLog
    from simod.settings.simod_settings import SimodSettings
    from simod.simod import Simod

    # Initialize 'output' folder and read configuration file
    output = Path("<path>/<to>/<outputs>/<folder>")
    configuration_path = Path("<path>/<to>/<configuration>.yml")
    settings = SimodSettings.from_path(configuration_path)

    # Read and preprocess event log
    event_log = EventLog.from_path(
        log_ids=settings.common.log_ids,
        train_log_path=settings.common.train_log_path,
        test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
        need_test_partition=settings.common.perform_final_evaluation,
    )

    # Instantiate and run SIMOD
    simod = Simod(settings=settings, event_log=event_log, output_dir=output)
    simod.run()

Modules Overview
----------------

Simod's codebase is organized into several key modules:

- **simod**: The main class that orchestrates the overall functionality.
- **settings**: Handles the parsing and validation of configuration files.
- **event_log**: Manages the IO operations of an event log as well as its preprocessing.
- **control_flow**: Utilities to discover and manage the control-flow model of a BPS model.
- **resource_model**: Utilities to discover and manage the resource model of a BPS model.
- **extraneous_delays**: Utilities to discover and manage the extraneous delays model of a BPS model.
- **simulation**: Manages the data model of a BPS model and its simulation and quality assessment.

Detailed Module Documentation
-----------------------------

Below is the detailed documentation for each module:

SIMOD class
^^^^^^^^^^^

.. automodule:: simod.simod
   :members:
   :undoc-members:
   :exclude-members: final_bps_model

Settings Module
^^^^^^^^^^^^^^^

SIMOD settings
""""""""""""""

.. automodule:: simod.settings.simod_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, common, preprocessing, control_flow, resource_model, extraneous_activity_delays, version

Common settings
"""""""""""""""

.. automodule:: simod.settings.common_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, train_log_path, log_ids, test_log_path, process_model_path, perform_final_evaluation, num_final_evaluations, evaluation_metrics, use_observed_arrival_distribution, clean_intermediate_files, discover_data_attributes, DL, TWO_GRAM_DISTANCE, THREE_GRAM_DISTANCE, CIRCADIAN_EMD, CIRCADIAN_WORKFORCE_EMD, ARRIVAL_EMD, RELATIVE_EMD, ABSOLUTE_EMD, CYCLE_TIME_EMD

Preprocessing settings
""""""""""""""""""""""

.. automodule:: simod.settings.preprocessing_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, multitasking, enable_time_concurrency_threshold, concurrency_thresholds

Control-flow model settings
"""""""""""""""""""""""""""

.. automodule:: simod.settings.control_flow_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, SPLIT_MINER_V1, SPLIT_MINER_V2, optimization_metric, num_iterations, num_evaluations_per_iteration, gateway_probabilities, mining_algorithm, epsilon, eta, discover_branch_rules, f_score, replace_or_joins, prioritize_parallelism

Resource model settings
"""""""""""""""""""""""

.. automodule:: simod.settings.resource_model_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, optimization_metric, num_iterations, num_evaluations_per_iteration, discovery_type, granularity, confidence, support, participation, discover_prioritization_rules, discover_batching_rules, fuzzy_angle

Extraneous delays settings
""""""""""""""""""""""""""

.. automodule:: simod.settings.extraneous_delays_settings
   :members:
   :undoc-members:
   :exclude-members: model_config, optimization_metric, discovery_method, num_iterations, num_evaluations_per_iteration

Event Log Module
^^^^^^^^^^^^^^^^

.. automodule:: simod.event_log.event_log
   :members:
   :undoc-members:
   :exclude-members: write_xes, train_partition, validation_partition, train_validation_partition, test_partition, log_ids, process_name

.. automodule:: simod.event_log.preprocessor
   :members:
   :undoc-members:
   :exclude-members: MultitaskingSettings, Settings

Control-flow Model Module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simod.control_flow.settings
   :members:
   :undoc-members:
   :exclude-members: output_dir, provided_model_path, project_name, optimization_metric, gateway_probabilities_method, mining_algorithm, epsilon, eta, replace_or_joins, prioritize_parallelism, f_score, from_hyperopt_dict

.. automodule:: simod.control_flow.optimizer
   :members:
   :undoc-members:
   :exclude-members: event_log, initial_bps_model, settings, base_directory, best_bps_model, evaluation_measurements, cleanup

.. automodule:: simod.control_flow.discovery
   :members:
   :undoc-members:
   :exclude-members: add_bpmn_diagram_to_model, SplitMinerV1Settings, SplitMinerV2Settings, discover_process_model_with_split_miner_v1, discover_process_model_with_split_miner_v2

Resource Model Module
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simod.resource_model.settings
   :members:
   :undoc-members:
   :exclude-members: output_dir, process_model_path, project_name, optimization_metric, calendar_discovery_params, discover_prioritization_rules, discover_batching_rules, from_hyperopt_dict

.. automodule:: simod.resource_model.optimizer
   :members:
   :undoc-members:
   :exclude-members: event_log, initial_bps_model, settings, base_directory, best_bps_model, evaluation_measurements, cleanup

Extraneous Delays Model Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simod.extraneous_delays.optimizer
   :members:
   :undoc-members:
   :exclude-members: cleanup

.. automodule:: simod.extraneous_delays.types
   :members:
   :undoc-members:
   :exclude-members: activity_name, delay_id, duration_distribution

.. automodule:: simod.extraneous_delays.utilities
   :members:
   :undoc-members:
   :exclude-members:

Simulation Module
^^^^^^^^^^^^^^^^^

.. automodule:: simod.simulation.parameters.BPS_model
   :members:
   :undoc-members:
   :exclude-members: process_model, gateway_probabilities, case_arrival_model, resource_model, extraneous_delays, case_attributes, global_attributes, event_attributes, prioritization_rules, batching_rules, branch_rules, calendar_granularity

.. automodule:: simod.simulation.prosimos
   :members:
   :undoc-members:
   :exclude-members: simulate_in_parallel, evaluate_logs, bpmn_path, parameters_path, output_log_path, num_simulation_cases, simulation_start

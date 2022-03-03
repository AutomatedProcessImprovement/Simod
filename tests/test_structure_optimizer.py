import os
from pathlib import Path

from simod.configuration import config_data_from_file, Configuration
from simod.event_log import LogReader
from simod.structure_optimizer import StructureOptimizer

optimize_config_files = [
    'optimize_debug_config.yml',
    # 'optimize_debug_with_model_config.yml',
    # 'optimize_debug_config_2.yml',
]

# NOTE: This is a very slow test which is already executed inside test_cli.py. So, we comment this out for now until
# we separate unit tests from system and acceptance tests.
#
# def test_best_parameters(entry_point):
#     for path in optimize_config_files:
#         config_path = Path(os.path.join(entry_point, path))
#         config = config_data_from_file(config_path)
#         config_structure = config.pop('strc')
#         config.pop('tm')
#         config_structure.update(config)
#         config_structure = Configuration(**config_structure)
#         config_structure.fill_in_derived_fields()
#
#         log = LogReader(config['log_path'])
#
#         structure_optimizer = StructureOptimizer(config_structure, log)
#         structure_optimizer.run()
#
#         assert structure_optimizer.best_output is not None
#         assert structure_optimizer.best_parameters is not None
#         assert structure_optimizer.best_parameters['gate_management'] is not None

from simod.configuration import Configuration
from simod.event_log import LogReader
from simod.structure_optimizer import StructureOptimizer

optimize_config_files = [
    'optimize_debug_config.yml',
    # 'optimize_debug_with_model_config.yml',
    # 'optimize_debug_config_2.yml',
]

# NOTE: This is a very slow test which is already executed inside test_cli.py. So, we comment this out for now until
# we separate unit tests from system and integration tests.
#
# def test_best_parameters(entry_point):
#     for path in optimize_config_files:
#         config_path = Path(os.path.join(entry_point, path))
#         config = config_data_from_file(config_path)
#         config_structure = config.pop('strc')
#         config.pop('tm')
#         config_structure.update(config)
#         config_structure = Configuration(**config_structure)
#
#         log = LogReader(config['log_path'])
#
#         structure_optimizer = StructureOptimizer(config_structure, log)
#         structure_optimizer.run()
#
#         assert structure_optimizer.best_output is not None
#         assert structure_optimizer.best_parameters is not None
#         assert structure_optimizer.best_parameters['gate_management'] is not None


structure_config_yaml = """
max_eval_s: 2
concurrency:
- 0.0
- 1.0
epsilon:
- 0.0
- 1.0
eta:
- 0.0
- 1.0
gate_management:
- equiprobable
- discovery
or_rep:
- true
- false
and_prior:
- true
- false
"""


def test_StructureOptimizer(entry_point):
    config = Configuration.from_yaml_str(structure_config_yaml)

    config.log_path = entry_point / 'PurchasingExample.xes'

    log_reader = LogReader(config.log_path)

    optimizer = StructureOptimizer(config, log_reader)
    optimizer.run()

    assert optimizer.best_parameters is not None

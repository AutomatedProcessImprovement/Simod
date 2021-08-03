import os
from pathlib import Path

from simod.common_routines import extract_structure_parameters

from .configuration import Configuration
from .decorators import safe_exec_with_values_and_status, timeit
from .stochastic_miner_datatypes import BPMNGraph
from .structure_optimizer import StructureOptimizer
from .writers import xml_writer


# class StructureOptimizerForStochasticProcessMiner(StructureOptimizer):
#     bpmn_graph: BPMNGraph
#     discover_model: bool
#
#     def __init__(self, settings: Configuration, log, discover_model: bool):
#         super().__init__(settings, log)
#         self.discover_model = discover_model
#
#     @timeit(rec_name='EXTRACTING_PARAMS')
#     @safe_exec_with_values_and_status
#     def _extract_parameters(self, settings: dict, structure, parameters, **kwargs) -> None:
#         if isinstance(settings, dict):
#             settings = Configuration(**settings)
#
#         _, process_graph = structure
#         num_inst = len(self.log_valdn.caseid.unique())  # TODO: shouldn't we use self.log_train here?
#         start_time = self.log_valdn.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")  # getting minimum date
#
#         if self.discover_model:
#             # SplitMiner output path
#             model_path = Path(os.path.join(settings.output, settings.project_name + '.bpmn'))
#         else:
#             # Provided model path
#             model_path = self.settings.model_path
#         self.bpmn_graph = BPMNGraph.from_bpmn_path(model_path)
#
#         structure_parameters = extract_structure_parameters(
#             settings=settings, process_graph=process_graph, log=self.log_train, model_path=model_path)
#
#         parameters = {**parameters, **{'resource_pool': structure_parameters.resource_pool,
#                                        'time_table': structure_parameters.time_table,
#                                        'arrival_rate': structure_parameters.arrival_rate,
#                                        'sequences': structure_parameters.sequences,
#                                        'elements_data': structure_parameters.elements_data,
#                                        'instances': num_inst,
#                                        'start_time': start_time}}
#         bpmn_path = os.path.join(settings.output, settings.project_name + '.bpmn')
#         xml_writer.print_parameters(bpmn_path, bpmn_path, parameters)
#
#         self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
#         self.log_valdn['source'] = 'log'
#         self.log_valdn['run_num'] = 0
#         self.log_valdn['role'] = 'SYSTEM'
#         self.log_valdn = self.log_valdn[~self.log_valdn.task.isin(['Start', 'End'])]

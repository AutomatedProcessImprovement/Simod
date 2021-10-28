import os
import platform as pl
import shutil
import subprocess

from networkx import DiGraph

from .cli_formatter import print_subsection, print_step, print_asset
from .configuration import Configuration, MiningAlgorithm
from .decorators import safe_exec
from .readers import bpmn_reader as br
from .readers import process_structure


class StructureMiner:
    """This class extracts all the BPS parameters"""
    bpmn: br.BpmnReader
    process_graph: DiGraph
    settings: Configuration
    is_safe: bool

    def __init__(self, settings: Configuration):
        self.is_safe = True
        self.settings = settings

    def execute_pipeline(self) -> None:
        if self.settings.model_path is None:
            print_subsection("Mining the model structure")
            log_path = self.settings.output / (self.settings.project_name + '.xes')
            print_asset(f"Log file {log_path}")
            self.is_safe = self._mining_structure(is_safe=self.is_safe)
        else:
            print_step("Copying the model")
            shutil.copy(self.settings.model_path, self.settings.output)

        model_path = self.settings.output / (self.settings.project_name + '.bpmn')
        self.bpmn = br.BpmnReader(model_path)
        self.process_graph = process_structure.create_process_structure(self.bpmn)

    @safe_exec
    def _mining_structure(self, **kwargs) -> None:
        miner = self._get_miner(self.settings.mining_alg)
        miner(self.settings)

    def _get_miner(self, miner: MiningAlgorithm):
        if miner is MiningAlgorithm.SM1:
            return self._sm1_miner
        elif miner is MiningAlgorithm.SM2:
            return self._sm2_miner
        elif miner is MiningAlgorithm.SM3:
            return self._sm3_miner
        else:
            raise ValueError(miner)

    @staticmethod
    def _sm2_miner(settings: Configuration):
        """
        Executes splitminer for bpmn structure mining.
        """
        # Event log file_name
        file_name = settings.project_name
        input_route = os.path.join(settings.output, file_name + '.xes')
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.append('-Xmx2G')
        args.extend(['-cp',
                     (settings.sm2_path.__str__() + sep + os.path.join(os.path.dirname(settings.sm2_path), 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SM2',
                     input_route,
                     os.path.join(settings.output, file_name),
                     str(settings.concurrency)])
        subprocess.call(args)

    @staticmethod
    def _sm1_miner(settings: Configuration) -> None:
        """
        Executes splitminer for bpmn structure mining.
        """
        # Event log file_name
        file_name = settings.project_name
        input_route = os.path.join(settings.output, file_name + '.xes')
        # Mining structure definition
        args = ['java', '-jar', settings.sm1_path,
                str(settings.epsilon), str(settings.eta),
                input_route,
                os.path.join(settings.output, file_name)]
        subprocess.call(args)

    @staticmethod
    def _sm3_miner(settings: Configuration) -> None:
        """
        Executes splitminer for bpmn structure mining.
        """
        # Event log file_name
        file_name = settings.project_name
        input_route = os.path.join(settings.output, file_name + '.xes')
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.extend(['-Xmx2G', '-Xms1024M'])
        args.extend(['-cp',
                     (settings.sm3_path.__str__() + sep + os.path.join(os.path.dirname(settings.sm3_path), 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SMD',
                     str(settings.epsilon), str(settings.eta),
                     # TODO: in some cases .and_prior and .or_rep are lists
                     str(settings.and_prior), str(settings.or_rep), 'false',
                     input_route,
                     os.path.join(settings.output, file_name)])
        subprocess.call(args)

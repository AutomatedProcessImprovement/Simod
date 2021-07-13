import os
import platform as pl
import shutil
import subprocess

from .readers import bpmn_reader as br
from .readers import process_structure as gph

from .cli_formatter import *
from .configuration import Configuration, MiningAlgorithm
from .decorators import safe_exec
from .log_repairing.conformance_checking import evaluate_alignment


class StructureMiner:
    """
    This class extracts all the BPS parameters
    """

    def __init__(self, settings: Configuration, log):
        self.log = log
        self.is_safe = True
        self.settings = settings

    def execute_pipeline(self) -> None:
        """
        Main method for mining structure
        """
        print_subsection("Split Mining")
        if self.settings.model_path is None:
            self.is_safe = self._mining_structure(is_safe=self.is_safe)
        else:
            print_step("Copying The Model")
            shutil.copy(self.settings.model_path, self.settings.output)
        print_subsection("Alignment Evaluation")
        self.is_safe = self._evaluate_alignment(is_safe=self.is_safe, model_path=self.settings.model_path)

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

        Returns
        -------
        None
            DESCRIPTION.
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
                     (settings.sm2_path.__str__() + sep + os.path.join('external_tools', 'splitminer2', 'lib', '*')),
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

        Returns
        -------
        None
            DESCRIPTION.
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
    
        Returns
        -------
        None
            DESCRIPTION.
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
                     (settings.sm3_path.__str__() + sep + os.path.join('external_tools', 'splitminer3', 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SMD',
                     str(settings.epsilon), str(settings.eta),
                     'false', 'false', 'false',
                     input_route,
                     os.path.join(settings.output, file_name)])
        subprocess.call(args)

    @safe_exec
    def _evaluate_alignment(self, **kwargs) -> None:
        """
        Evaluates alignment
        Returns
        -------
        None
            DESCRIPTION.

        """
        self.bpmn = br.BpmnReader(os.path.join(self.settings.output, self.settings.project_name + '.bpmn'))
        self.process_graph = gph.create_process_structure(self.bpmn)
        evaluate_alignment(self.process_graph, self.log, self.settings)

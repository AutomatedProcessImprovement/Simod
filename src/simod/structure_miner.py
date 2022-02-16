import os
import platform as pl
import shutil
import subprocess
from pathlib import Path

from networkx import DiGraph

from .cli_formatter import print_subsection, print_step, print_asset
from .configuration import Configuration, MiningAlgorithm
from .readers import bpmn_reader as br
from .readers import process_structure


class StructureMiner:
    """This class extracts all the BPS parameters"""
    bpmn: br.BpmnReader
    process_graph: DiGraph
    _xes_path: Path
    _settings: Configuration

    def __init__(self, xes_path: Path, settings: Configuration):
        self._xes_path = xes_path
        self._settings = settings

    def execute_pipeline(self):
        if self._settings.model_path is None:
            print_subsection("Mining the model structure")
            print_asset(f"Log file {self._xes_path}")
            self._mining_structure(self._xes_path)
        else:
            print_step("Copying the model")
            shutil.copy(self._settings.model_path, self._settings.output)

        model_path = self._settings.output / (self._settings.project_name + '.bpmn')
        self.bpmn = br.BpmnReader(model_path)
        self.process_graph = process_structure.create_process_structure(self.bpmn)

    def _mining_structure(self, xes_path: Path):
        miner = self._get_miner(self._settings.mining_alg)
        miner(xes_path, self._settings)

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
    def _sm2_miner(xes_path: Path, settings: Configuration):
        """
        Executes SplitMiner2 for BPMN structure mining.
        """
        file_name = settings.project_name
        sep = ';' if pl.system().lower() == 'windows' else ':'
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.append('-Xmx2G')
        args.extend(['-cp',
                     (settings.sm2_path.__str__() + sep + os.path.join(os.path.dirname(settings.sm2_path), 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SM2',
                     str(xes_path),
                     os.path.join(settings.output, file_name),
                     str(settings.concurrency)])
        subprocess.call(args)

    @staticmethod
    def _sm1_miner(xes_path: Path, settings: Configuration):
        """
        Executes SplitMiner2 for BPMN structure mining.
        """
        file_name = settings.project_name
        args = ['java', '-jar', settings.sm1_path,
                str(settings.epsilon), str(settings.eta),
                str(xes_path),
                os.path.join(settings.output, file_name)]
        subprocess.call(args)

    @staticmethod
    def _sm3_miner(xes_path: Path, settings: Configuration):
        """
        Executes SplitMiner3 for BPMN structure mining.
        """
        # Event log file_name
        file_name = settings.project_name
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.extend(['-Xmx2G', '-Xms1024M'])

        if isinstance(settings.and_prior, list):
            and_prior_setting = str([str(value) for value in settings.and_prior])
        else:
            and_prior_setting = str(settings.and_prior)

        if isinstance(settings.or_rep, list):
            or_rep_setting = str([str(value) for value in settings.or_rep])
        else:
            or_rep_setting = str(settings.or_rep)

        args.extend(['-cp',
                     (settings.sm3_path.__str__() + sep + os.path.join(os.path.dirname(settings.sm3_path), 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SMD',
                     str(settings.epsilon), str(settings.eta),
                     and_prior_setting, or_rep_setting, 'false',
                     str(xes_path),
                     os.path.join(settings.output, file_name)])
        subprocess.call(args)

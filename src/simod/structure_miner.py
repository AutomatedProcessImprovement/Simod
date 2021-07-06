import os
import platform as pl
import subprocess

import readers.bpmn_reader as br
import readers.process_structure as gph

from .decorators import safe_exec
from .log_repairing.conformance_checking import evaluate_alignment


class StructureMiner():
    """
    This class extracts all the BPS parameters
    """
    def __init__(self, settings, log):
        """constructor"""
        self.log = log
        self.is_safe = True
        self.settings = settings

    def execute_pipeline(self) -> None:
        """
        Main method for mining structure
        """
        self.is_safe = self._mining_structure(is_safe=self.is_safe)
        # TODO: is=f a model is provided, start here right away
        self.is_safe = self._evaluate_alignment(is_safe=self.is_safe)

    @safe_exec
    def _mining_structure(self, **kwargs) -> None:
        miner = self._get_miner(self.settings['mining_alg'])
        miner(self.settings)

    def _get_miner(self, miner):
        if miner == 'sm1':
            return self._sm1_miner
        elif miner == 'sm2':
            return self._sm2_miner
        elif miner == 'sm3':
            return self._sm3_miner
        else:
            raise ValueError(miner)

    @staticmethod
    def _sm2_miner(settings):
        """
        Executes splitminer for bpmn structure mining.

        Returns
        -------
        None
            DESCRIPTION.
        """
        print(" -- Mining Process Structure --")
        # Event log file_name
        file_name = settings['file'].split('.')[0]
        input_route = os.path.join(settings['output'], file_name + '.xes')
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.append('-Xmx2G')
        args.extend(['-cp',
                     (settings['sm2_path'] + sep + os.path.join(
                         'external_tools', 'splitminer2', 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SM2',
                     input_route,
                     os.path.join(settings['output'], file_name),
                     str(settings['concurrency'])])
        subprocess.call(args)

    @staticmethod
    def _sm1_miner(settings) -> None:
        """
        Executes splitminer for bpmn structure mining.

        Returns
        -------
        None
            DESCRIPTION.
        """
        print(" -- Mining Process Structure --")
        # Event log file_name
        file_name = settings['file'].split('.')[0]
        input_route = os.path.join(settings['output'], file_name + '.xes')
        # Mining structure definition
        args = ['java', '-jar', settings['sm1_path'],
                str(settings['epsilon']), str(settings['eta']),
                input_route,
                os.path.join(settings['output'], file_name)]
        subprocess.call(args)

    @staticmethod
    def _sm3_miner(settings) -> None:
        """
        Executes splitminer for bpmn structure mining.
    
        Returns
        -------
        None
            DESCRIPTION.
        """
        print(" -- Mining Process Structure --")
        # Event log file_name
        file_name = settings['file'].split('.')[0]
        input_route = os.path.join(settings['output'], file_name + '.xes')
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if not pl.system().lower() == 'windows':
            args.extend(['-Xmx2G', '-Xms1024M'])
        args.extend(['-cp',
                     (settings['sm3_path'] + sep + os.path.join(
                         'external_tools', 'splitminer3', 'lib', '*')),
                     'au.edu.unimelb.services.ServiceProvider',
                     'SMD',
                     str(settings['epsilon']), str(settings['eta']),
                     'false', 'false', 'false',
                     input_route,
                     os.path.join(settings['output'], file_name)])
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
        # load bpmn model
        self.bpmn = br.BpmnReader(os.path.join(
            self.settings['output'],
            self.settings['file'].split('.')[0] + '.bpmn'))
        self.process_graph = gph.create_process_structure(self.bpmn)
        # Evaluate alignment

        evaluate_alignment(self.process_graph,
                           self.log,
                           self.settings)

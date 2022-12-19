import os
import platform as pl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from simod.cli_formatter import print_warning, print_step
from simod.configuration import PROJECT_DIR, StructureMiningAlgorithm, GatewayProbabilitiesDiscoveryMethod

sm1_path: Path = PROJECT_DIR / 'external_tools/splitminer/splitminer.jar'
sm2_path: Path = PROJECT_DIR / 'external_tools/splitminer2/sm2.jar'
sm3_path: Path = PROJECT_DIR / 'external_tools/splitminer3/bpmtk.jar'


@dataclass
class Settings:
    """
    Settings for the structure miner.
    """

    gateway_probabilities_method: GatewayProbabilitiesDiscoveryMethod
    mining_algorithm: StructureMiningAlgorithm = StructureMiningAlgorithm.SPLIT_MINER_3

    # Split Miner 1 and 3
    epsilon: Optional[float] = None
    eta: Optional[float] = None

    # Split Miner 2
    concurrency: Optional[float] = 0.0

    # Split Miner 3
    prioritize_parallelism: Optional[bool] = False
    replace_or_joins: Optional[bool] = False

    # Private
    _sm1_path: Path = sm1_path
    _sm2_path: Path = sm2_path
    _sm3_path: Path = sm3_path

    @staticmethod
    def default() -> 'Settings':
        return Settings(
            gateway_probabilities_method=GatewayProbabilitiesDiscoveryMethod.DISCOVERY,
            mining_algorithm=StructureMiningAlgorithm.SPLIT_MINER_3,
            prioritize_parallelism=False,
            replace_or_joins=False,
            epsilon=0.5,
            eta=0.5,
        )

    @staticmethod
    def from_stream(stream: Union[str, bytes]) -> Optional['Settings']:
        settings = yaml.load(stream, Loader=yaml.FullLoader)

        if 'structure_optimizer' in settings:
            settings = settings['structure_optimizer']

        gateway_probabilities_method = settings.get('gateway_probabilities_method', None)
        if gateway_probabilities_method is not None:
            gateway_probabilities_method = GatewayProbabilitiesDiscoveryMethod.from_str(gateway_probabilities_method)

        mining_algorithm = settings.get('mining_algorithm', None)
        if mining_algorithm is None:
            mining_algorithm = settings.get('mining_alg', None)  # legacy key support
        if mining_algorithm is not None:
            mining_algorithm = StructureMiningAlgorithm.from_str(mining_algorithm)
        if mining_algorithm is None:
            print_warning('No mining algorithm specified.')
            return None

        epsilon = settings.get('epsilon', None)
        assert type(epsilon) is not list, 'epsilon must be a single value'

        eta = settings.get('eta', None)
        assert type(eta) is not list, 'eta must be a single value'

        concurrency = settings.get('concurrency', 0.0)
        assert type(concurrency) is not list, 'concurrency must be a single value'

        prioritize_parallelism = None
        parallelism = settings.get('prioritize_parallelism', None)
        if parallelism is not None:
            if isinstance(parallelism, str):
                prioritize_parallelism = [parallelism.lower() == 'true']
            elif isinstance(parallelism, list):
                prioritize_parallelism = parallelism
            else:
                raise ValueError('prioritize_parallelism must be a list or a string.')

        replace_or_joins = None
        or_joins = settings.get('replace_or_joins', None)
        if or_joins is not None:
            if isinstance(or_joins, str):
                replace_or_joins = [or_joins.lower() == 'true']
            elif isinstance(or_joins, list):
                replace_or_joins = or_joins
            else:
                raise ValueError('replace_or_joins must be a list or a string.')

        return Settings(
            gateway_probabilities_method=gateway_probabilities_method,
            mining_algorithm=mining_algorithm,
            epsilon=epsilon,
            eta=eta,
            concurrency=concurrency,
            prioritize_parallelism=prioritize_parallelism,
            replace_or_joins=replace_or_joins
        )

    def to_dict(self) -> dict:
        return {
            'mining_algorithm': self.mining_algorithm.value if self.mining_algorithm else None,
            'gateway_probabilities_method': self.gateway_probabilities_method.value if self.gateway_probabilities_method else None,
            'epsilon': self.epsilon,
            'eta': self.eta,
            'concurrency': self.concurrency,
            'prioritize_parallelism': self.prioritize_parallelism,
            'replace_or_joins': self.replace_or_joins
        }


class StructureMiner:
    """
    Discovers the process structure from an event log file in XES format.
    """

    def __init__(
            self,
            mining_algorithm: StructureMiningAlgorithm,
            xes_path: Path,
            output_model_path: Path,
            concurrency: Optional[float] = None,
            eta: Optional[float] = None,
            epsilon: Optional[float] = None,
            prioritize_parallelism: Optional[bool] = None,
            replace_or_joins: Optional[bool] = None,
    ):
        global sm2_path, sm3_path

        self.xes_path = xes_path
        self.output_model_path = output_model_path
        self.mining_algorithm = mining_algorithm

        if mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_2:
            self.concurrency = concurrency
            self.split_miner_path = sm2_path
        elif mining_algorithm is StructureMiningAlgorithm.SPLIT_MINER_3:
            self.eta = eta
            self.epsilon = epsilon
            self.prioritize_parallelism = prioritize_parallelism
            self.replace_or_joins = replace_or_joins
            self.split_miner_path = sm3_path
        else:
            raise ValueError(f'Unknown mining algorithm: {mining_algorithm}')

    def run(self):
        miner = self.mining_algorithm

        if miner is StructureMiningAlgorithm.SPLIT_MINER_1:
            raise NotImplementedError('Split Miner 1 is not supported anymore.')
        elif miner is StructureMiningAlgorithm.SPLIT_MINER_2:
            self._sm2_miner(self.xes_path, self.split_miner_path, self.concurrency)
        elif miner is StructureMiningAlgorithm.SPLIT_MINER_3:
            self._sm3_miner(self.xes_path, self.split_miner_path, self.eta, self.epsilon, self.prioritize_parallelism,
                            self.replace_or_joins)
        else:
            raise ValueError(f'Unknown mining algorithm: {miner}')

        assert self.output_model_path.exists(), f"Model file {self.output_model_path} hasn't been mined"

    def _model_path_without_suffix(self) -> Path:
        if self.output_model_path is not None:
            return self.output_model_path.with_suffix('')
        else:
            raise ValueError('No output model path specified.')

    def _sm1_miner(self, xes_path: Path, settings: Settings):
        output_path = str(self._model_path_without_suffix())
        args = [
            'java', '-jar', settings._sm1_path,
            str(settings.epsilon),
            str(settings.eta),
            str(xes_path),
            output_path
        ]

        print_step(f'SplitMiner1 is running with the following arguments: {args}')
        subprocess.call(args)

    def _sm2_miner(self, xes_path: Path, sm2_path: Path, concurrency: float):
        output_path = str(self._model_path_without_suffix())
        sep = ';' if pl.system().lower() == 'windows' else ':'
        args = ['java']
        if pl.system().lower() != 'windows':
            args.append('-Xmx2G')
        args.extend(
            ['-cp',
             (sm2_path.__str__() + sep + os.path.join(os.path.dirname(sm2_path), 'lib', '*')),
             'au.edu.unimelb.services.ServiceProvider',
             'SM2',
             str(xes_path),
             output_path,
             str(concurrency)]
        )

        print_step(f'SplitMiner2 is running with the following arguments: {args}')
        subprocess.call(args)

    def _sm3_miner(
            self,
            xes_path: Path,
            sm3_path: Path,
            eta: float,
            epsilon: float,
            prioritize_parallelism: bool,
            replace_or_joins: bool,
    ):
        output_path = str(self._model_path_without_suffix())
        sep = ';' if pl.system().lower() == 'windows' else ':'

        args = ['java']

        if pl.system().lower() != 'windows':
            args.extend(['-Xmx2G', '-Xms1024M'])

        # prioritizes parallelism on loops
        parallelism_first = str(prioritize_parallelism).lower()
        # replaces non trivial OR joins
        replace_or_joins = str(replace_or_joins).lower()
        # removes loop activity markers (false increases model complexity)
        remove_loop_activity_markers = 'false'

        args.extend([
            '-cp',
            (sm3_path.__str__() + sep + os.path.join(os.path.dirname(sm3_path), 'lib', '*')),
            'au.edu.unimelb.services.ServiceProvider',
            'SMD',
            str(eta),
            str(epsilon),
            parallelism_first,
            replace_or_joins,
            remove_loop_activity_markers,
            str(xes_path),
            output_path
        ])

        print_step(f'SplitMiner3 is running with the following arguments: {args}')
        subprocess.call(args)

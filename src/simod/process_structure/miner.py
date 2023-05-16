import os
from pathlib import Path
from typing import Optional

from simod.cli_formatter import print_step
from simod.settings.control_flow_settings import (
    ProcessModelDiscoveryAlgorithm,
)
from simod.settings.simod_settings import PROJECT_DIR
from simod.utilities import is_windows, execute_external_command

sm2_path: Path = PROJECT_DIR / "external_tools/splitminer2/sm2.jar"
sm3_path: Path = PROJECT_DIR / "external_tools/splitminer3/bpmtk.jar"


class StructureMiner:
    """
    Discovers the process structure from an event log file in XES format.
    """

    def __init__(
        self,
        mining_algorithm: ProcessModelDiscoveryAlgorithm,
        xes_path: Path,
        output_model_path: Path,
        concurrency: Optional[float] = None,
        eta: Optional[float] = None,
        epsilon: Optional[float] = None,
        prioritize_parallelism: Optional[bool] = None,
        replace_or_joins: Optional[bool] = None,
    ):
        self.xes_path = xes_path
        self.output_model_path = output_model_path
        self.mining_algorithm = mining_algorithm

        if mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
            self.concurrency = concurrency
        elif mining_algorithm is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
            self.eta = eta
            self.epsilon = epsilon
            self.prioritize_parallelism = prioritize_parallelism
            self.replace_or_joins = replace_or_joins
        else:
            raise ValueError(f"Unknown mining algorithm: {mining_algorithm}")

    def run(self):
        miner = self.mining_algorithm

        if miner is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_2:
            self._sm2_miner(self.xes_path, self.concurrency)
        elif miner is ProcessModelDiscoveryAlgorithm.SPLIT_MINER_3:
            self._sm3_miner(
                self.xes_path,
                self.eta,
                self.epsilon,
                self.prioritize_parallelism,
                self.replace_or_joins,
            )
        else:
            raise ValueError(f"Unknown mining algorithm: {miner}")

        assert (
            self.output_model_path.exists()
        ), f"Model file {self.output_model_path} hasn't been mined"

    def _model_path_without_suffix(self) -> Path:
        if self.output_model_path is not None:
            return self.output_model_path.with_suffix("")
        else:
            raise ValueError("No output model path specified.")

    def _sm2_miner(self, xes_path: Path, concurrency: float):
        # Define args depending on the system is running
        if is_windows():
            # Windows: ';' as separator and escape string with '"'
            args = ["java"]
            split_miner_path = "\"" + sm2_path.__str__() + ";" + os.path.join(os.path.dirname(sm2_path), "lib", "*") + "\""
            input_log_path = "\"" + str(xes_path) + "\""
            model_output_path = "\"" + str(self._model_path_without_suffix()) + "\""
        else:
            # Linux: ':' as separator and add memory specs
            args = ["java", "-Xmx2G", "-Xms1024M"]
            split_miner_path = sm2_path.__str__() + ":" + os.path.join(os.path.dirname(sm2_path), "lib", "*")
            input_log_path = str(xes_path)
            model_output_path = str(self._model_path_without_suffix())
        # Prepare command structure
        args += [
            "-cp",
            split_miner_path,
            "au.edu.unimelb.services.ServiceProvider",
            "SM2",
            input_log_path,
            model_output_path,
            str(concurrency),
        ]
        # Execute command
        print_step(f"SplitMiner2 is running with the following arguments: {args}")
        execute_external_command(args)

    def _sm3_miner(
        self,
        xes_path: Path,
        eta: float,
        epsilon: float,
        prioritize_parallelism: bool,
        replace_or_joins: bool,
    ):
        # Define args depending on the system is running
        if is_windows():
            # Windows: ';' as separator and escape string with '"'
            args = ["java"]
            split_miner_path = "\"" + sm3_path.__str__() + ";" + os.path.join(os.path.dirname(sm3_path), "lib", "*") + "\""
            input_log_path = "\"" + str(xes_path) + "\""
            model_output_path = "\"" + str(self._model_path_without_suffix()) + "\""
        else:
            # Linux: ':' as separator and add memory specs
            args = ["java", "-Xmx2G", "-Xms1024M"]
            split_miner_path = sm3_path.__str__() + ":" + os.path.join(os.path.dirname(sm3_path), "lib", "*")
            input_log_path = str(xes_path)
            model_output_path = str(self._model_path_without_suffix())
        # Prepare command structure
        args += [
            "-cp",
            split_miner_path,
            "au.edu.unimelb.services.ServiceProvider",
            "SMD",
            str(eta),
            str(epsilon),
            str(prioritize_parallelism).lower(),  # Prioritize parallelism over loops
            str(replace_or_joins).lower(),  # Replace non-trivial OR joins
            "false",  # Remove loop activity markers (false increases model complexity)
            input_log_path,
            model_output_path,
        ]
        # Execute command
        print_step(f"SplitMiner3 is running with the following arguments: {args}")
        execute_external_command(args)

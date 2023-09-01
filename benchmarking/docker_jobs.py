import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Simod configuration string to generate the configuration file for each experiment
configuration_str = Path("input/config_f_naive.yml").read_text()

# Path to the directory containing the logs and configuration files in the Docker container
container_input_dir = Path("/usr/src/Simod/input")

# TODO: use file paths relative to the configuration file
# TODO: remove Xvfb
# TODO: update running Simod via CLI


@dataclass
class Experiment:
    """
    Dataclass representing an experiment with a train and, optionally, test log, and the corresponding configuration.
    """

    train_log_path: Path
    test_log_path: Optional[Path] = None

    _assets_dir: Path = Path("input")
    _configuration_dir: Path = Path("input/configs")
    _logs_dir: Path = Path("input/logs")
    _experiments_output_dir: Path = Path("outputs")
    _simod_version: str = "3.6.0"

    @property
    def assets_dir(self) -> Path:
        """
        Directory containing the assets (logs, configurations, etc.) for the experiment on the host machine.
        """
        return self._assets_dir

    @property
    def experiments_output_dir(self) -> Path:
        """
        Directory containing the outputs of the experiments on the host machine.
        """
        return self._experiments_output_dir

    @property
    def configuration_path(self) -> Path:
        """
        Path to the configuration file for the experiment on the host machine.
        """
        return self._configuration_dir / f"{self.train_log_path.stem}.yml"

    @property
    def simod_version(self) -> str:
        """
        Version of Simod to use for the experiment.
        """
        return self._simod_version

    def __post_init__(self):
        assert self._assets_dir.exists()
        assert self._logs_dir.exists()
        self._configuration_dir.mkdir(exist_ok=True, parents=True)
        self._generate_configuration()

    def _generate_configuration(self):
        """
        Update the configuration with the paths to the logs and save it to the configuration directory
        on the host machine.
        """
        global container_input_dir, configuration_str

        config = configuration_str.replace(
            "<train_log_path>", str((container_input_dir / "logs" / self.train_log_path.name).absolute())
        )
        if self.test_log_path:
            config = config.replace(
                "<test_log_path>", str((container_input_dir / "logs" / self.test_log_path.name).absolute())
            )

        self.configuration_path.write_text(config)


def main():
    experiments = [
        Experiment(
            train_log_path=Path("logs/AcademicCredentials_train.csv.gz"),
            test_log_path=Path("logs/AcademicCredentials_test.csv.gz"),
        ),
        Experiment(
            train_log_path=Path("logs/BPIC_2012_train.csv.gz"),
            test_log_path=Path("logs/BPIC_2012_test.csv.gz"),
        ),
        Experiment(
            train_log_path=Path("logs/BPIC_2017_train.csv.gz"),
            test_log_path=Path("logs/BPIC_2017_test.csv.gz"),
        ),
        Experiment(
            train_log_path=Path("logs/CallCenter_train.csv.gz"),
            test_log_path=Path("logs/CallCenter_test.csv.gz"),
        ),
    ]

    with Path("timing.txt").open("a") as f:
        for experiment in experiments:
            print(f"Running {experiment.configuration_path}")
            start = time.time()
            run_with_docker(experiment)
            duration = time.time() - start
            f.write(f"{experiment.configuration_path.stem}: {duration}\n")
            # flushing saves each record to disk (not guaranteed though) without waiting
            # for other experiments to finish in case they crash
            f.flush()
            print(f"Finished {experiment.configuration_path} in {duration} seconds")


def run_with_docker(experiment: Experiment):
    global container_input_dir

    docker_run_script = f"""#!/bin/bash

cd /usr/src/Simod
Xvfb :99 &>/dev/null & disown
poetry run simod optimize --config_path {container_input_dir}/configs/{experiment.configuration_path.name}
"""
    docker_run_script_path = experiment.assets_dir / "docker_run.sh"
    docker_run_script_path.write_text(docker_run_script)

    docker_run_script_path_in_container = container_input_dir / "docker_run.sh"
    cmd = [
        "docker",
        "run",
        "-v",
        f"{experiment.assets_dir.absolute()}:/usr/src/Simod/input",
        "-v",
        f"{experiment.experiments_output_dir.absolute()}:/usr/src/Simod/outputs",
        f"nokal/simod:{experiment.simod_version}",
        "bash",
        docker_run_script_path_in_container.absolute(),
    ]

    with open(f"{experiment.configuration_path.stem}.out", "w") as f:
        with open(f"{experiment.configuration_path.stem}.err", "w") as g:
            subprocess.run(cmd, stdout=f, stderr=g)


def create_slurm_job_script(jobs_dir: Path, config_path: Path, prefix: Optional[str] = None):
    """
    Create a SLURM job script for the given configuration file to run in the HPC.
    """

    job_name = config_path.stem
    partition = "main"
    nodes = 1
    cpus_per_task = 20
    mem = "40G"
    time = "24:00:00"

    script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --mail-user=ihar.suvorau@ut.ee
#SBATCH --mail-type=END,FAIL

module load any/jdk/1.8.0_265
module load py-xvfbwrapper
source /gpfs/space/home/suvorau/simod_v3.2.0_prerelease_2/Simod/venv/bin/activate
xvfb-run simod optimize --config_path {config_path.absolute()}
"""

    job_script = jobs_dir / f"{job_name}{prefix}.sh"

    with open(job_script, "w") as f:
        f.write(script)

    return job_script


if __name__ == "__main__":
    main()

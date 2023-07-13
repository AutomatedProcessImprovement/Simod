import subprocess
import time
from pathlib import Path
from typing import Optional

config_str = Path("input/config.yml").read_text()


def create_configuration_file(
    log_path: Path,
    config_base: str,
    host_config_dir: Path,
    container_input_dir: Path,
    test_log_path: Optional[Path] = None,
    prefix: Optional[str] = None,
):
    """
    Creates a configuration file
    """
    config = config_base.replace("<train_log_path>", str((container_input_dir / "logs" / log_path.name).absolute()))
    if test_log_path:
        config = config.replace("<test_log_path>", str((container_input_dir / "logs" / test_log_path.name).absolute()))

    if prefix:
        config_path = (host_config_dir / (Path(log_path).stem + prefix)).with_suffix(".yml")
    else:
        config_path = (host_config_dir / Path(log_path).stem).with_suffix(".yml")

    with open(config_path, "w") as f:
        f.write(config)

    return config_path


def create_job_script(jobs_dir: Path, config_path: Path, prefix: Optional[str] = None):
    """Create a job script"""

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


def run_docker_job(config_path: Path, host_input_dir: Path, host_output_dir: Path):
    """Run a job in a Docker container"""
    container_base_path = Path("/usr/src/Simod/input")

    docker_run_script = f"""#!/bin/bash

cd /usr/src/Simod
Xvfb :99 &>/dev/null & disown
poetry run simod optimize --config_path {container_base_path}/configs/{config_path.name}
"""

    docker_run_script_path = host_input_dir / "docker_run.sh"
    docker_run_container_script_path = container_base_path / "docker_run.sh"

    with open(docker_run_script_path, "w") as f:
        f.write(docker_run_script)

    cmd = [
        "docker",
        "run",
        "-v",
        f"{host_input_dir.absolute()}:/usr/src/Simod/input",
        "-v",
        f"{host_output_dir.absolute()}:/usr/src/Simod/outputs",
        "nokal/simod",
        "bash",
        docker_run_container_script_path.absolute(),
    ]

    with open(f"{config_path.stem}.out", "w") as f:
        with open(f"{config_path.stem}.err", "w") as g:
            subprocess.run(cmd, stdout=f, stderr=g)


def main():
    container_input_dir = Path("/usr/src/Simod/input")

    host_input_dir = Path("input")
    config_dir = host_input_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    host_output_dir = Path("outputs")
    host_output_dir.mkdir(exist_ok=True, parents=True)

    log_paths = [
        # (Path("logs/AcademicCredentials_train.csv.gz"), Path("logs/AcademicCredentials_test.csv.gz")),
        (Path("logs/BPIC_2012_train.csv.gz"), Path("logs/BPIC_2012_test.csv.gz")),
        (Path("logs/BPIC_2017_train.csv.gz"), Path("logs/BPIC_2017_test.csv.gz")),
        # (Path("logs/CallCenter_train.csv.gz"), Path("logs/CallCenter_test.csv.gz")),
    ]

    config_paths = [
        create_configuration_file(
            host_input_dir / log_path[0],
            config_str,
            config_dir,
            container_input_dir,
            test_log_path=host_input_dir / log_path[1],
        )
        for log_path in log_paths
    ]

    timing_file = Path("timing.txt")

    with open(timing_file, "a") as f:
        for config_path in config_paths:
            print(f"Running {config_path}")
            start = time.time()
            run_docker_job(config_path, host_input_dir, host_output_dir)
            duration = time.time() - start
            f.write(f"{config_path.stem}: {duration}\n")
            print(f"Finished {config_path} in {duration} seconds")


if __name__ == "__main__":
    main()

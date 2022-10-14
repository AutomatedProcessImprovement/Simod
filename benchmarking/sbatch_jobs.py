import subprocess
from pathlib import Path

config_str = """
version: 2
common:
  log_path: assets/Production.xes
  exec_mode: optimizer
  repetitions: 5
  simulation: true
  evaluation_metrics: 
    - dl
    - day_hour_emd
    - log_mae
    - mae
preprocessing:
  multitasking: false
structure:
  max_evaluations: 40
  mining_algorithm: sm3
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gateway_probabilities:
    - equiprobable
    - discovery
  or_rep:
    - true
    - false
  and_prior:
    - true
    - false
calendars:
  max_evaluations: 20
  case_arrival:
    discovery_type: undifferentiated
    granularity: 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation: 0.4
  resource_profiles:
    discovery_type: pool
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""


def create_configuration_file(log_path: Path, config_base: str, config_dir: Path):
    """Create a configuration file"""
    config = config_base.replace('assets/Production.xes', str(log_path.absolute()))
    config_path = (config_dir / Path(log_path).stem).with_suffix('.yml')
    with open(config_path, 'w') as f:
        f.write(config)
    return config_path


def create_job_script(jobs_dir: Path, config_path: Path):
    """Create a job script"""

    job_name = config_path.stem
    partition = 'main'
    nodes = 1
    cpus_per_task = 20
    mem = '40G'
    time = '24:00:00'

    script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}

module load any/jdk/1.8.0_265
module load py-xvfbwrapper
source venv/bin/activate
xvfb-run simod optimize --config_path {config_path.absolute()}
"""

    job_script = jobs_dir / f'{job_name}.sh'
    with open(job_script, 'w') as f:
        f.write(script)

    return job_script


def submit_job(job_script: Path):
    """Submit a job to the SLURM scheduler"""
    cmd = ['sbatch', job_script]
    subprocess.run(cmd)


def main():
    log_paths = [
        Path('logs/confidential_1000.xes'),
        Path('logs/confidential_2000.xes'),
        Path('logs/cvs_pharmacy.xes'),
        Path('logs/BPI_Challenge_2012_W_Two_TS.xes'),
        Path('logs/BPI_Challenge_2017_W_Two_TS.xes'),
        Path('logs/PurchasingExample.xes'),
        Path('logs/Production.xes'),
        Path('logs/ConsultaDataMining201618.xes'),
        Path('logs/insurance.xes'),
        Path('logs/Application-to-Approval-Government-Agency.xes'),
    ]

    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)

    config_paths = [create_configuration_file(log_path, config_str, config_dir) for log_path in log_paths]

    jobs_dir = Path('jobs')
    jobs_dir.mkdir(exist_ok=True)

    job_paths = [create_job_script(jobs_dir, config_path) for config_path in config_paths]

    for job_path in job_paths:
        submit_job(job_path)


if __name__ == '__main__':
    main()

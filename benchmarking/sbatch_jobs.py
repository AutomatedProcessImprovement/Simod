import subprocess
from pathlib import Path
from typing import Optional

config_str = """
version: 2
common:
  log_path: assets/train_Production.xes
  test_log_path: assets/test_Production.xes
  exec_mode: optimizer
  log_ids:
    case: case_id
    activity: Activity
    resource: Resource
    start_time: start_time
    end_time: end_time
  repetitions: 5
  simulation: true
  evaluation_metrics: 
    - dl
    - circadian_emd
    - absolute_hourly_emd
    - cycle_time_emd
preprocessing:
  multitasking: false
structure:
  max_evaluations: 40
  optimization_metric: dl
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
  max_evaluations: 40
  optimization_metric: absolute_hourly_emd
  resource_profiles:
    discovery_type: differentiated
    granularity: 
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""


def create_configuration_file(
        log_path: Path,
        config_base: str,
        config_dir: Path,
        test_log_path: Optional[Path] = None,
        prefix: Optional[str] = None):
    """
    Creates a configuration file
    """
    config = config_base.replace('assets/train_Production.xes', str(log_path.absolute()))
    if test_log_path:
        config = config.replace('assets/test_Production.xes', str(test_log_path.absolute()))

    config_path = (config_dir / (Path(log_path).stem + prefix)).with_suffix('.yml')

    with open(config_path, 'w') as f:
        f.write(config)

    return config_path


def create_job_script(jobs_dir: Path, config_path: Path, prefix: Optional[str] = None):
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
#SBATCH --mail-user=ihar.suvorau@ut.ee
#SBATCH --mail-type=END,FAIL

module load any/jdk/1.8.0_265
module load py-xvfbwrapper
source /gpfs/space/home/suvorau/simod_v3.2.0_prerelease_2/Simod/venv/bin/activate
xvfb-run simod optimize --config_path {config_path.absolute()}
"""

    job_script = jobs_dir / f'{job_name}{prefix}.sh'

    with open(job_script, 'w') as f:
        f.write(script)

    return job_script


def submit_job(job_script: Path):
    """Submit a job to the SLURM scheduler"""
    cmd = ['sbatch', job_script]
    subprocess.run(cmd)


def main():
    prefix = '_differentiated_absolute-hourly-emd'

    log_paths = [
        (Path('logs/BPIC_2012_W_contained_train.csv'), Path('logs/BPIC_2012_W_contained_test.csv')),
        (Path('logs/BPIC_2017_W_contained_train.csv'), Path('logs/BPIC_2017_W_contained_test.csv')),
        (Path('logs/ConsultaDataMining201618_train.csv'), Path('logs/ConsultaDataMining201618_test.csv')),
        (Path('logs/Governmental_Agency_train.csv'), Path('logs/Governmental_Agency_test.csv')),
        (Path('logs/poc_processmining_train.csv'), Path('logs/poc_processmining_test.csv')),
        (Path('logs/Production_train.csv'), Path('logs/Production_test.csv')),
    ]

    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)

    config_paths = [
        create_configuration_file(log_path[0], config_str, config_dir, test_log_path=log_path[1], prefix=prefix)
        for log_path in log_paths
    ]

    jobs_dir = Path('jobs')
    jobs_dir.mkdir(exist_ok=True)

    job_paths = [
        create_job_script(jobs_dir, config_path, prefix=prefix)
        for config_path in config_paths
    ]

    for job_path in job_paths:
        submit_job(job_path)


if __name__ == '__main__':
    main()

import subprocess
import time
from pathlib import Path
from typing import Optional

config_str = """
version: 2
common:
  log_path: assets/train_Production.xes
  test_log_path: assets/test_Production.xes
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
  extraneous_activity_delays: true
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
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
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
        host_config_dir: Path,
        container_input_dir: Path,
        test_log_path: Optional[Path] = None,
        prefix: Optional[str] = None):
    """
    Creates a configuration file
    """
    config = config_base.replace(
        'assets/train_Production.xes',
        str((container_input_dir / 'logs' / log_path.name).absolute())
    )
    if test_log_path:
        config = config.replace(
            'assets/test_Production.xes',
            str((container_input_dir / 'logs' / test_log_path.name).absolute())
        )

    config_path = (host_config_dir / (Path(log_path).stem + prefix)).with_suffix('.yml')

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


def run_docker_job(config_path: Path, host_input_dir: Path, host_output_dir: Path):
    """Run a job in a Docker container"""
    container_base_path = Path('/usr/src/Simod/input')

    docker_run_script = f"""#!/bin/bash

cd /usr/src/Simod
source venv/bin/activate
Xvfb :99 &>/dev/null & disown
simod optimize --config_path {container_base_path}/configs/{config_path.name}
"""

    docker_run_script_path = host_input_dir / 'docker_run.sh'
    docker_run_container_script_path = container_base_path / 'docker_run.sh'

    with open(docker_run_script_path, 'w') as f:
        f.write(docker_run_script)

    cmd = [
        'docker', 'run',
        '-v', f'{host_input_dir.absolute()}:/usr/src/Simod/input',
        '-v', f'{host_output_dir.absolute()}:/usr/src/Simod/outputs',
        'nokal/simod',
        'bash', docker_run_container_script_path.absolute()
    ]

    with open(f'{config_path.stem}.out', 'w') as f:
        with open(f'{config_path.stem}.err', 'w') as g:
            subprocess.run(cmd, stdout=f, stderr=g)


def main():
    prefix = '_differentiated_absolute-hourly-emd'

    container_input_dir = Path('/usr/src/Simod/input')

    host_input_dir = Path('input')
    config_dir = host_input_dir / 'configs'
    config_dir.mkdir(exist_ok=True, parents=True)

    host_output_dir = Path('outputs')
    host_output_dir.mkdir(exist_ok=True, parents=True)

    log_paths = [
        (Path('logs/Production_train.csv'), Path('logs/Production_test.csv')),
        # (Path('logs/ConsultaDataMining201618_train.csv'), Path('logs/ConsultaDataMining201618_test.csv')),
        # (Path('logs/BPIC_2012_W_contained_train.csv'), Path('logs/BPIC_2012_W_contained_test.csv')),
        # (Path('logs/BPIC_2017_W_contained_train.csv'), Path('logs/BPIC_2017_W_contained_test.csv')),
        # (Path('logs/poc_processmining_train.csv'), Path('logs/poc_processmining_test.csv')),
        # (Path('logs/Governmental_Agency_train.csv'), Path('logs/Governmental_Agency_test.csv')),
    ]

    config_paths = [
        create_configuration_file(
            host_input_dir / log_path[0],
            config_str,
            config_dir,
            container_input_dir,
            test_log_path=host_input_dir / log_path[1],
            prefix=prefix
        )
        for log_path in log_paths
    ]

    timing_file = Path('timing.txt')

    with open(timing_file, 'a') as f:
        for config_path in config_paths:
            print(f'Running {config_path}')
            start = time.time()
            run_docker_job(config_path, host_input_dir, host_output_dir)
            duration = time.time() - start
            f.write(f'{config_path.stem}: {duration}\n')
            print(f'Finished {config_path} in {duration} seconds')


if __name__ == '__main__':
    main()

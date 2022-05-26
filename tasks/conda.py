import json

from invoke import task, run
from conda.cli.python_api import run_command, Commands


@task
def create_env(ctx, name='simod'):
    """
    Create conda environment.
    """
    ctx.run(f'conda env create -f environment.yml -n {name}')


def env_path(name='simod') -> str:
    """
    Get conda environment path.
    """
    result = run(f'conda --envs --json')
    config = json.loads(result.stdout)
    try:
        envs = config['envs']
        for env_path in envs:
            if env_path.split('/')[-1] == name:
                return env_path
    except KeyError:
        raise KeyError('No environment found.')


@task
def dump_requirements(ctx, name='simod'):
    """
    Dump requirements.txt.
    """
    # conda_env_path = env_path(name)
    run(f'pip list --format=freeze > requirements.txt')
    run(f'conda env export -n {name} > environment.yml')  # TODO: doesn't work

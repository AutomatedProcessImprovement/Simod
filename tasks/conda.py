from invoke import task


@task
def create_env(ctx, name='simod'):
    """
    Create conda environment.
    """
    ctx.run(f'conda env create -f environment.yml -n {name}')

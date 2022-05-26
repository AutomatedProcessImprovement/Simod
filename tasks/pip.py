from invoke import task


@task
def create_env(ctx):
    """
    Create pip environment in venv.
    """
    ctx.run('python3 -m venv venv')
    ctx.run('venv/bin/pip install --upgrade pip')
    ctx.run('venv/bin/pip install -r requirements.txt')

    with ctx.cd('external_tools/Prosimos'):
        ctx.run('venv/bin/pip install -e .')

    with ctx.cd('external_tools/pm4py-wrapper'):
        ctx.run('venv/bin/pip install -e .')

    ctx.run('venv/bin/pip install -e .')


@task
def dump_requirements(ctx):
    """
    Dump requirements.txt.
    """
    ctx.run('venv/bin/pip list --format=freeze > requirements.txt')
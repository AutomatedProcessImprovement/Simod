from invoke import task


@task
def unit(ctx):
    ctx.run("pytest -v -m 'not integration'")


@task
def integration(ctx):
    ctx.run("pytest -v -m 'integration'")
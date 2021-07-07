import click


def print_section(message: str):
    click.secho(f'\n{message}', bold=True)
    click.echo('=' * len(message))


def print_subsection(message: str):
    click.secho(f'\n{message}', bold=True)
    click.echo('-' * len(message))


def print_asset(message: str):
    click.echo(f'\n▶︎ {message}')


def print_message(message: str):
    click.echo(message.capitalize())


def print_notice(message: str):
    click.echo(f'● {message}')


def print_step(message: str):
    click.echo(f'➜ {message}')

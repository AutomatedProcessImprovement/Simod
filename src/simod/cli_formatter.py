import click


def print_section(message: str):
    click.secho(f"\n{message}", bold=True)
    click.secho("=" * len(message), bold=True)


def print_subsection(message: str):
    click.secho(f"\n{message}")
    click.echo("-" * len(message))


def print_asset(message: str):
    click.secho(f"\n︎{message}", bold=True)


def print_message(message: str):
    click.echo(message)


def print_notice(message: str):
    click.secho(f"\n{message}", bold=True)


def print_warning(message: str):
    click.secho(f"\n{message}", bold=True)


def print_step(message: str):
    click.echo(f"\n{message}")

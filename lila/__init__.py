import click

from .generate_environment import generate_environment
from .compute_score import compute_metric

@click.group()
def cli():
    """Lila Legibility Command Line Interface"""
    pass


cli.add_command(generate_environment)
cli.add_command(compute_metric)
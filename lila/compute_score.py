import click
import importlib


@click.command()
@click.option(
    "--metric-name",
    "metric",
    help=("The metric to use. "
    "It should be inside an importable module and the name is specified "
    "using `:`, e.g. `lila.metrics.dragan:legibility`."),
)
@click.option("--environment", help="The environment to evaluate the metric in.")
def compute_metric(metric, environment) -> float:

    return 42

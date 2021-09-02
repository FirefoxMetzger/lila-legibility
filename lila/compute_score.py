import click



known_metrics = {

}



@click.command()
@click.option("--metric-name", "metric")
@click.option("--environment")
def compute_metric(metric, environment) -> float:


    return 42
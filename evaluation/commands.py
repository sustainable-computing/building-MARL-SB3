import typer
import numpy as np

from buildingenvs import BuildingEnvStrings
from evaluation.online.evaluate import evaluate_policies
from evaluation.offline.commands import app as ope_app

app = typer.Typer()
app.add_typer(ope_app, name="offline")


@app.command("online")
def evaluate(
    building_env: BuildingEnvStrings = typer.Option(
                BuildingEnvStrings.denver.value,
                help="The building environment to train agents on"),
    building_config_loc: str = typer.Option(
        "configs/buildingconfig/building_denver.yaml",
        help="The location of the building config file"),
    zone: str = typer.Option(None),
    energy_plus_loc: str = typer.Option(...),
    policy_library_path: str = typer.Option(...),
    policy_type: str = typer.Option(...),
    init_policy_log_std: float = typer.Option(np.log(0.1)),
    init_policy_log_std_path: str = typer.Option(""),
    num_days: int = typer.Option(...),
    start_day: int = typer.Option(1),
    start_month: int = typer.Option(1),
    start_year: int = typer.Option(1991),
    save_path: str = typer.Option("data/policy_evaluation/brute_force/"),
    parallelize: bool = typer.Option(False),
    max_cpu_cores: int = typer.Option(1),
    seed: int = typer.Option(1337),
):
    if hasattr(policy_type, "value"):
        policy_type = policy_type.value
    evaluate_policies(**locals())

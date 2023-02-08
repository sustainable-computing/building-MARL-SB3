import numpy as np
from ope.evaluate import evaluate
from ope import OPEMethodStrings
from policies import PolicyTypeStrings

import typer
from typing import List


app = typer.Typer()


@app.command("list-methods")
def list_methods():
    for method in OPEMethodStrings:
        print(method.value)


@app.command("list-policy-types")
def list_policy_types():
    for policy_type in PolicyTypeStrings:
        print(policy_type.value)


@app.command("evaluate")
def evaluate_ope(methods: List[OPEMethodStrings] = typer.Argument(
                    ..., help="OPE methods to evaluate"),
                 log_data_path: str = typer.Option(...),
                 policy_library_path: str = typer.Option(...),
                 policy_type: PolicyTypeStrings = typer.Option(...),
                 init_policy_log_std: float = typer.Option(np.log(0.1)),
                 init_policy_log_std_path: str = typer.Option(""),
                 num_days: int = typer.Option(...),
                 use_behavior_p_score: bool = typer.Option(...),
                 behavior_policy_path: str = typer.Option(""),
                 save_path: str = typer.Option("data/policy_evaluation/ope/"),
                 parallelize: bool = typer.Option(False),
                 max_cpu_cores: int = typer.Option(1)):
    methods = [method.value for method in methods]
    evaluate(**locals())

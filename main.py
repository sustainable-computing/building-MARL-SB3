import typer
from train import app as train_app
from evaluation.commands import app as eval_app
from tuneparams.tune_ppo import app as tune_ppo


app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="evaluation")
app.add_typer(tune_ppo, name="tuneparams")


if __name__ == "__main__":
    app()

import typer
from train.ppo_train import app as ppo_app


app = typer.Typer()
app.add_typer(ppo_app, name="ppo")

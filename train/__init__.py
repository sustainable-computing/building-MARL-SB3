import typer
from train.ppo.commands import app as ppo_app


app = typer.Typer()
app.add_typer(ppo_app, name="ppo")

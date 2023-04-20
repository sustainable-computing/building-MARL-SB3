import typer
import numpy as np

from evaluation.offline.commands import app as offline_eval_app
from evaluation.online.commands import app as online_eval_app

app = typer.Typer()
app.add_typer(offline_eval_app, name="offline")
app.add_typer(online_eval_app, name="online")

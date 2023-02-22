import typer
from train import app as train_app
from ope.commands import app as ope_app
from evaluation.commands import app as eval_app


app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(ope_app, name="ope")
app.add_typer(eval_app, name="evaluate")


if __name__ == "__main__":
    app()

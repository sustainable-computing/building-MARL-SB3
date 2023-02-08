import typer
from train import app as train_app
from ope.commands import app as ope_app


app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(ope_app, name="ope")


if __name__ == "__main__":
    app()

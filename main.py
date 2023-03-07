import typer
from train import app as train_app
from evaluation.commands import app as eval_app


app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="evaluation")


if __name__ == "__main__":
    app()

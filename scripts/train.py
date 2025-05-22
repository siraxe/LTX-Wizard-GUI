#!/usr/bin/env python

"""
Train LTXV models using configuration from YAML files.

This script provides a command-line interface for training LTXV models using
either LoRA fine-tuning or full model fine-tuning. It loads configuration from
a YAML file and passes it to the trainer.

Basic usage:
    train.py --config configs/ltxv_lora_config.yaml
"""

from pathlib import Path

import typer
import yaml
from rich.console import Console

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Train LTXV models using configuration from YAML files.",
)

@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    num_processes: int = typer.Option(
        1,
        "--num_processes",
        help="Number of processes (GPUs) to use. Note: This argument is for informational purposes within the script and does not override the actual launch configuration set by accelerate environment variables or accelerate launch.",
    ),
) -> None:
    """Train the model using the provided configuration file."""
    # Load the configuration from the YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Remove the 'misc' section if it exists to avoid validation errors
    if 'misc' in config_data:
        del config_data['misc']

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process
    trainer = LtxvTrainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    app()

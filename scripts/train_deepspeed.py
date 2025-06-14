#!/usr/bin/env python

"""
Train LTXV models using DeepSpeed for multi-GPU training.

This script provides a command-line interface for training LTXV models using
DeepSpeed. It loads configuration from a YAML file and passes it to the trainer.

This script is intended to be launched with the `deepspeed` command.

Basic usage:
    deepspeed train_deepspeed.py --deepspeed_config deepspeed_config.json --config configs/ltxv_lora_config.yaml
"""

import argparse
from pathlib import Path
import sys

import typer
import yaml
from rich.console import Console
import deepspeed # Import deepspeed

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer_deep import LtxvDeepSpeedTrainer # Import DeepSpeed trainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Train LTXV models using DeepSpeed for multi-GPU training.",
)

@app.command()
def main(
    config_path: str = typer.Option(..., "--config", help="Path to YAML configuration file"),
    deepspeed_config: str = typer.Option(
        ...,
        "--deepspeed_config",
        help="Path to DeepSpeed configuration JSON file.",
    ),
) -> None:
    """Train the model using the provided configuration file and DeepSpeed."""
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

    # Add DeepSpeed config path to the trainer config
    if 'deepspeed' not in config_data:
        config_data['deepspeed'] = {}
    config_data['deepspeed']['config_path'] = deepspeed_config

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process with DeepSpeed trainer
    trainer = LtxvDeepSpeedTrainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    # DeepSpeed handles argument parsing for its own arguments,
    # so we need to parse them separately and pass only the relevant ones to Typer.

    # Create a parser for DeepSpeed-specific arguments
    ds_parser = argparse.ArgumentParser(add_help=False)
    ds_parser.add_argument("--local_rank", type=int, default=-1)
    ds_parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    ds_parser.add_argument("--master_port", type=int, default=29500)
    ds_parser.add_argument("--node_rank", type=int, default=0)
    ds_parser.add_argument("--nproc_per_node", type=int, default=1)

    # Parse known DeepSpeed arguments from sys.argv
    # This will consume DeepSpeed arguments and leave others for the script's parser
    ds_args, remaining_argv_after_ds = ds_parser.parse_known_args()

    # Set the local_rank for DeepSpeed
    if ds_args.local_rank != -1:
        import os
        os.environ['LOCAL_RANK'] = str(ds_args.local_rank)

    # Create a parser for the script's own arguments (Typer's arguments)
    script_parser = argparse.ArgumentParser(add_help=False)
    script_parser.add_argument("--config", type=str, required=True)
    script_parser.add_argument("--deepspeed_config", type=str, required=True)

    # Now, parse the script's arguments from the remaining arguments
    # This will ensure Typer only sees what it expects
    script_args, final_remaining_argv = script_parser.parse_known_args(remaining_argv_after_ds)

    # Construct the arguments list for Typer
    # Typer expects arguments in the format it defines (e.g., --config value)
    typer_args = []
    if script_args.config:
        typer_args.extend(["--config", script_args.config])
    if script_args.deepspeed_config:
        typer_args.extend(["--deepspeed_config", script_args.deepspeed_config])

    # Pass the constructed arguments to the Typer app
    app(args=typer_args)

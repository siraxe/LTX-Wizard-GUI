#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating a new uv virtual environment..."
uv venv

echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Installing PyTorch nightly build externally..."
# Install PyTorch, torchvision, and torchaudio from the nightly build URL.
# These packages are explicitly excluded from uv.lock as per user's request.
uv pip install triton
uv pip install typer
uv pip install typing-extensions
uv pip install tzdata
uv pip install urllib3
uv pip install virtualenv
uv pip install zipp
echo "Synchronizing the virtual environment with uv.lock..."
# This will install all other dependencies listed in uv.lock into the venv.
uv sync

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "Deactivating the virtual environment..."
deactivate

echo "PyTorch nightly build has been installed externally, and the venv synchronized with uv.lock."
echo "You can now remove the virtual environment if it's no longer needed by running: rm -rf .venv"

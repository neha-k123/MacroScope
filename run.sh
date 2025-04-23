#!/bin/bash

# === CONFIG ===
VENV_PATH="../virtual-envs/macroVenv"  # REPLACE THIS with your venv path!
APP_PATH="./app.py"

# === ACTIVATE VENV ===
echo " Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# === VERIFY PYTHON ===
echo "Python in use: $(which python)"
echo "Pip in use:    $(which pip)"

# === INSTALL DEPENDENCIES (optional safety step) ===
pip install -r "./requirements.txt"

# === RUN MAIN SCRIPT ===
echo "Launching Streamlit dashboard..."
streamlit run "$APP_PATH"


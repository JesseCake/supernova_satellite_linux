#!/bin/bash
# This script activates the venv and runs the voice node. (assumes venv already set up)
# get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# activate the venv
source "$SCRIPT_DIR/.venv/bin/activate"
# run the voice node
python "$SCRIPT_DIR/satellite_client.py"

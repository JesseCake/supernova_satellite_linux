#!/bin/bash
# This script activates the venv and runs the voice node. (assumes venv already set up)
# get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# activate the venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Launcher relaunches on crash helpful as this node code is pretty rough in places
while true; do
    # run the voice node
    python "$SCRIPT_DIR/satellite_client.py"
    EXIT_CODE=$?

    # Don't restart on clean exit (eg CTRL C):
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo "[launcher] Clean exit, not restarting."
        break
    fi

    echo "[launcher] Crashed with exit code $EXIT_CODE, restarting in 3 seconds..."
    sleep 3
done

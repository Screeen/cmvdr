#!/bin/zsh

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" && pwd )"

# Project root = one folder up from script directory
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Change to the project root directory
# This ensures that the script runs from the project root
# and can find the main.py script and other resources correctly.
cd "$PROJECT_ROOT"

# Logs folder inside the project root
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"

# Timestamp for log filename (YYYY-MM-DD_HH-MM-SS)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Activate the virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Run the Python script and store logs in log/
PYTHONUNBUFFERED=1 python "$PROJECT_ROOT/main.py" --data_type synthetic 2>&1 | tee "$LOG_DIR/synthetic_$TIMESTAMP.log"

# Wait for user before closing
read -r "REPLY?Press enter to exit"
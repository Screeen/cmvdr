#!/bin/bash

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"

# Project root = one folder up from script directory
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Change to the project root directory
cd "$PROJECT_ROOT" || exit 1

# Logs folder inside the project root
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"

# Timestamp for log filename (YYYY-MM-DD_HH-MM-SS)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Activate the virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "Warning: virtualenv activate script not found at $PROJECT_ROOT/.venv/bin/activate"
fi

# Run the Python script and store logs in log/
PYTHONUNBUFFERED=1 python "$PROJECT_ROOT/main.py"  -c synthetic_demo 2>&1 | tee "$LOG_DIR/demo_$TIMESTAMP.log"

# Wait for any single key press before closing
read -n1 -s -r -p $'\nPress any key to exit...' KEY
echo

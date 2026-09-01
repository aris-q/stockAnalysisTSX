#!/bin/bash
# One command to set up (first time) and launch the app.
# Usage: ./run.sh
set -e

# Always operate relative to this script's own location, not wherever it's
# called from — so it works no matter where the repo was cloned/copied.
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "No venv found — creating one (first run only)..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Checking dependencies..."
pip install -q -r requirements.txt

echo "Starting app — open http://127.0.0.1:5090"
python3 app.py

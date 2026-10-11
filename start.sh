#!/usr/bin/env bash
# ---- AI Roleplayer: one-click launcher for macOS / Linux ----
set -e
cd "$(dirname "$0")"

echo "Setting up AI Roleplayer..."

# Create a virtual environment the first time.
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo
echo "Starting the app. Your browser will open shortly."
echo "(Press Ctrl+C to stop.)"
echo

python run.py

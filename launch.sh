#!/usr/bin/env bash

# Exit on error
set -e

# Determine repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🚀 Starting VideoSeal Web UI..."
echo "=========================================="

# Check Python installation
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python 3 is required but not found."
    exit 1
fi

VENV_DIR=".venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/update dependencies if needed
echo "🛠️ Checking dependencies..."
pip install -q -r requirements.txt

# Launch app with python
echo "🌐 Launching Gradio UI..."
python app.py

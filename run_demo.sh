#!/bin/bash
# Epistemic Flow Control - Demo Runner
#
# Usage: ./run_demo.sh

set -e

echo "💧 Epistemic Flow Control - Interactive Demo"
echo "============================================="
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "Installing demo dependencies..."
    pip install streamlit plotly pandas
fi

# Check if we're in the right directory
if [ ! -f "unified_system.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

echo ""
echo "Starting Streamlit demo..."
echo "Open http://localhost:8501 in your browser"
echo ""

streamlit run streamlit_demo/app.py

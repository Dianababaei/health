#!/bin/bash
# Artemis Health Dashboard Startup Script

echo "🐄 Starting Artemis Health Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "❌ Python is not installed."
    exit 1
fi

# Run the dashboard
echo "✅ Launching dashboard at http://localhost:8501"
echo ""
streamlit run dashboard/app.py

#!/bin/bash

# Quick start script for AI-Powered Pairs Trading Strategy

echo "=========================================="
echo "AI-Powered Pairs Trading Strategy"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
python3 scripts/setup_dirs.py

# Copy .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env file and add your API keys (optional for basic demo)"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo ""
echo "1. Basic demo (no API keys needed):"
echo "   python3 examples/usage_examples.py"
echo ""
echo "2. Run full pipeline:"
echo "   python3 src/main.py"
echo ""
echo "3. Add API keys to .env for full features:"
echo "   - ALPHA_VANTAGE_API_KEY (optional)"
echo "   - NEWS_API_KEY (optional)"
echo ""
echo "Documentation: See README.md"
echo ""

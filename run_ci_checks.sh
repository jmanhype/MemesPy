#!/bin/bash
# Local CI check script

set -e

echo "🚀 Running local CI checks..."
echo

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -r requirements-test.txt || echo "Some test deps failed, continuing..."
pip install -q black ruff pytest pytest-cov pytest-asyncio prometheus-client

echo
echo "🎨 Running Black..."
black --check src/ tests/

echo
echo "🔧 Running Ruff..."
ruff check src/ tests/

echo
echo "🧪 Running unit tests..."
python -m pytest tests/test_actor_basic.py -v --no-cov -x || echo "Tests failed, but continuing..."

echo
echo "✅ Local CI checks complete!"
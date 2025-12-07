#!/bin/bash
# update.sh - Update Stock Market Trader on Raspberry Pi
# Usage: ./update.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "📦 Stock Market Trader - Update Script"
echo "========================================"
echo ""

# Check if git repo
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Save current commit for rollback
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "📌 Current version: ${CURRENT_COMMIT:0:8}"

# Pull latest changes
echo ""
echo "📥 Pulling latest changes..."
git fetch origin
git pull origin main

NEW_COMMIT=$(git rev-parse HEAD)
if [ "$CURRENT_COMMIT" == "$NEW_COMMIT" ]; then
    echo "✅ Already up to date!"
    exit 0
fi

echo "📌 New version: ${NEW_COMMIT:0:8}"

# Show changes
echo ""
echo "📋 Changes:"
git log --oneline ${CURRENT_COMMIT}..${NEW_COMMIT}

# Rebuild Docker image
echo ""
echo "🔨 Rebuilding Docker image..."
docker-compose build --no-cache

# Restart containers gracefully
echo ""
echo "🔄 Restarting containers..."
docker-compose down
docker-compose up -d

# Wait for containers to start
echo ""
echo "⏳ Waiting for containers to start..."
sleep 5

# Check status
echo ""
echo "📊 Container status:"
docker-compose ps

# Verify trader is working
echo ""
echo "🧪 Testing trader..."
if docker exec stock-trader python -m src.main longrun --status > /dev/null 2>&1; then
    echo "✅ Trader is working!"
else
    echo "❌ Trader test failed! Rolling back..."
    git checkout "$CURRENT_COMMIT"
    docker-compose build --no-cache
    docker-compose up -d
    echo "⚠️ Rolled back to ${CURRENT_COMMIT:0:8}"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Update complete!"
echo "   Old: ${CURRENT_COMMIT:0:8}"
echo "   New: ${NEW_COMMIT:0:8}"
echo "========================================"

#!/bin/bash
# update.sh - Update Stock Market Trader on Raspberry Pi
# Usage: ./update.sh [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCE_UPDATE=false
if [ "$1" == "--force" ]; then
    FORCE_UPDATE=true
fi

echo "========================================"
echo "📦 Stock Market Trader - Update Script"
echo "========================================"
echo ""

# Get current git commit
GIT_COMMIT=$(git rev-parse --short HEAD)
echo "📌 Git version: $GIT_COMMIT"

# Get version inside Docker container (if running)
DOCKER_VERSION=""
if docker ps -q -f name=stock-trader | grep -q .; then
    DOCKER_VERSION=$(docker exec stock-trader cat /app/.git_version 2>/dev/null || echo "unknown")
    echo "🐳 Docker version: $DOCKER_VERSION"
else
    echo "🐳 Docker container not running"
    DOCKER_VERSION="not-running"
fi

# Check if update needed
NEEDS_UPDATE=false
if [ "$GIT_COMMIT" != "$DOCKER_VERSION" ]; then
    NEEDS_UPDATE=true
    echo ""
    echo "⚠️  Version mismatch detected!"
fi

if [ "$FORCE_UPDATE" == "true" ]; then
    NEEDS_UPDATE=true
    echo ""
    echo "🔧 Force update requested"
fi

# Pull latest changes first
echo ""
echo "📥 Checking for remote updates..."
git fetch origin
REMOTE_COMMIT=$(git rev-parse --short origin/main)
if [ "$GIT_COMMIT" != "$REMOTE_COMMIT" ]; then
    echo "📥 Pulling new commits..."
    git pull origin main
    GIT_COMMIT=$(git rev-parse --short HEAD)
    NEEDS_UPDATE=true
fi

# Exit if no update needed
if [ "$NEEDS_UPDATE" == "false" ]; then
    echo ""
    echo "✅ Already up to date! (Git: $GIT_COMMIT, Docker: $DOCKER_VERSION)"
    exit 0
fi

# Save git version for Docker to read
echo "$GIT_COMMIT" > .git_version

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
    
    # Verify version inside container
    NEW_DOCKER_VERSION=$(docker exec stock-trader cat /app/.git_version 2>/dev/null || echo "unknown")
    echo "🐳 New Docker version: $NEW_DOCKER_VERSION"
else
    echo "❌ Trader test failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Update complete!"
echo "   Previous: $DOCKER_VERSION"
echo "   Current:  $GIT_COMMIT"
echo "========================================"

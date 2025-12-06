#!/bin/bash
# Stock Market Analysis - Launch Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse arguments
MODE="${1:-cli}"

case "$MODE" in
    "cli")
        # Run CLI mode
        shift || true
        python -m src.main "$@"
        ;;
    "tui")
        # Run TUI mode
        cd tui
        if [ ! -d "node_modules" ]; then
            echo "Installing TUI dependencies..."
            if command -v bun &> /dev/null; then
                bun install
            else
                npm install
            fi
        fi
        
        if command -v bun &> /dev/null; then
            bun run src/App.tsx
        else
            npx tsx src/App.tsx
        fi
        ;;
    "scan")
        # Quick scan
        python -m src.main scan --quick
        ;;
    "api")
        # API mode for TUI
        python -m src.main api
        ;;
    "install")
        # Install all dependencies
        echo "Installing Python dependencies..."
        pip install -r requirements.txt
        
        echo "Installing TUI dependencies..."
        cd tui
        if command -v bun &> /dev/null; then
            bun install
        else
            npm install
        fi
        echo "Installation complete!"
        ;;
    *)
        echo "Usage: $0 [cli|tui|scan|api|install] [options]"
        echo ""
        echo "Modes:"
        echo "  cli      Run command-line interface (default)"
        echo "  tui      Run rich terminal UI"
        echo "  scan     Quick market scan"
        echo "  api      Run API server for TUI"
        echo "  install  Install all dependencies"
        echo ""
        echo "CLI Examples:"
        echo "  $0 cli analyze AAPL"
        echo "  $0 cli scan --quick"
        echo "  $0 cli backtest TSLA"
        ;;
esac

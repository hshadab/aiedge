#!/bin/bash

# AI Edge → AP2 Payment Demo Runner
# This script starts all three services:
# 1. Rust backend (port 3001) - ZKML prover & policy engine
# 2. AP2 Python service (port 3002) - Google AP2 protocol
# 3. Frontend server (port 3000) - Demo UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  AI Edge → AP2 Payment Demo"
echo "========================================"
echo ""
echo "Services:"
echo "  • Frontend:    http://localhost:3000"
echo "  • Backend:     http://localhost:3001"
echo "  • AP2 Service: http://localhost:3002"
echo ""

# Track PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Stopping services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Also kill any orphaned processes on our ports
    lsof -ti:3000 2>/dev/null | xargs -r kill 2>/dev/null || true
    lsof -ti:3001 2>/dev/null | xargs -r kill 2>/dev/null || true
    lsof -ti:3002 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup INT TERM

# 1. Start AP2 Python service
if lsof -Pi :3002 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✓ AP2 service already running on port 3002"
else
    echo "Starting AP2 service..."
    cd "$DEMO_DIR/ap2-service"

    # Check for virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    # Install dependencies if needed
    if [ ! -f ".deps-installed" ]; then
        echo "  Installing Python dependencies..."
        pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt 2>/dev/null
        touch .deps-installed
    fi

    python main.py &
    PIDS+=($!)
    echo "  AP2 service starting (PID: ${PIDS[-1]})"
    sleep 2
fi

# 2. Start Rust backend
if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✓ Backend already running on port 3001"
else
    echo "Starting backend server..."
    cd "$DEMO_DIR/backend"

    # Check for release build
    if [ ! -f "target/release/ai-edge-ap2-demo" ]; then
        echo "  Building backend (first run may take a while)..."
        cargo build --release 2>&1 | head -20
    fi

    cargo run --release &
    PIDS+=($!)
    echo "  Backend starting (PID: ${PIDS[-1]})"
    sleep 3
fi

# 3. Start frontend server
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✓ Frontend already running on port 3000"
else
    echo "Starting frontend server..."
    cd "$DEMO_DIR/frontend"
    python3 -m http.server 3000 &
    PIDS+=($!)
    echo "  Frontend starting (PID: ${PIDS[-1]})"
fi

echo ""
echo "========================================"
echo "  Demo is running!"
echo "========================================"
echo ""
echo "  Open http://localhost:3000 in your browser"
echo ""
echo "  Services:"
echo "    Frontend:    http://localhost:3000"
echo "    Backend API: http://localhost:3001"
echo "    AP2 API:     http://localhost:3002"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
wait

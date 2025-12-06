#!/usr/bin/env bash

################################################################################
# run_backend.sh — Mock Backend Application Launcher
#
# Purpose: Start the FastAPI application with proper environment setup
#
# Usage:   ./scripts/run_backend.sh [--port PORT] [--host HOST]
#
# This script:
#   1. Verifies virtual environment exists and is activated
#   2. Loads .env.dev environment variables
#   3. Checks critical dependencies
#   4. Creates required directories (logs, data)
#   5. Verifies port is available
#   6. Starts Uvicorn server with auto-reload (development mode)
#   7. Logs output to logs/backend.log
#   8. Handles graceful shutdown
#
# Environment: Mock (development mode with auto-reload by default)
#
################################################################################

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'  # No Color

# Default configuration
HOST="0.0.0.0"
PORT="8000"
RELOAD=true
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/backend.log"
PID_FILE="${LOG_DIR}/backend.pid"

# Parse command-line arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --port)
        PORT="$2"
        shift 2
        ;;
      --host)
        HOST="$2"
        shift 2
        ;;
      --no-reload)
        RELOAD=false
        shift
        ;;
      *)
        print_error "Unknown option: $1"
        print_usage
        exit 1
        ;;
    esac
  done
}

# Helper functions
print_header() {
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║          RAG ADMIN PIPELINE - BACKEND LAUNCHER (MOCK)          ║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo
}

print_ok() {
  echo -e "${GREEN}✓ $*${NC}"
}

print_info() {
  echo -e "${YELLOW}$*${NC}"
}

print_error() {
  echo -e "${RED}✗ $*${NC}"
}

print_usage() {
  echo "Usage: ./scripts/run_backend.sh [--port PORT] [--host HOST] [--no-reload]"
}

# Step 1: Verify virtual environment
check_venv() {
  print_info "[1/6] Checking virtual environment ..."
  
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    print_error "Virtual environment not activated"
    print_info "Run: source venv/bin/activate"
    exit 1
  fi
  
  print_ok "Virtual environment is active: $VIRTUAL_ENV"
  echo
}

# Step 2: Load environment variables
load_env_dev() {
  print_info "[2/6] Loading environment variables from .env.dev ..."
  
  if [ -f ".env.dev" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env.dev
    set +a
    print_ok ".env.dev loaded"
  else
    print_info ".env.dev not found. Using defaults."
  fi
  
  # Set required env vars if not already set
  export ENVIRONMENT="${ENVIRONMENT:-mock}"
  export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  
  echo
}

# Step 3: Check critical dependencies
check_dependencies() {
  print_info "[3/6] Checking critical dependencies ..."
  
  local deps=("fastapi" "uvicorn" "pydantic")
  local missing=0
  
  for dep in "${deps[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
      print_ok "$dep"
    else
      print_error "$dep not found"
      missing=$((missing + 1))
    fi
  done
  
  if [ $missing -gt 0 ]; then
    print_error "$missing dependency/dependencies missing"
    print_info "Run: ./scripts/setup_env.sh"
    exit 1
  fi
  
  echo
}

# Step 4: Create required directories
create_directories() {
  print_info "[4/6] Creating required directories ..."
  
  mkdir -p "$LOG_DIR"
  mkdir -p "data"
  mkdir -p "data/cache"
  mkdir -p "data/uploads"
  mkdir -p "data/temp"
  
  print_ok "Directories created"
  echo
}

# Step 5: Check port availability
check_port() {
  print_info "[5/6] Checking if port $PORT is available ..."
  
  if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
      print_error "Port $PORT is already in use"
      print_info "Use: ./scripts/run_backend.sh --port 8001"
      exit 1
    fi
  else
    print_info "lsof not available, skipping port check"
  fi
  
  print_ok "Port $PORT is available"
  echo
}

# Step 6: Start application
start_application() {
  print_info "[6/6] Verifying application module ..."
  
  if ! python -c "from src.api.main import app" 2>/dev/null; then
    print_error "Failed to import src.api.main:app"
    print_error "Check: src/api/main.py exists and has no syntax errors"
    exit 1
  fi
  
  print_ok "Application verified"
  echo
}

# Print startup information
print_startup_info() {
  echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║                  🚀 STARTING APPLICATION                        ║${NC}"
  echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo
  echo -e "${BLUE}Server Configuration:${NC}"
  echo "  • Host: $HOST"
  echo "  • Port: $PORT"
  echo "  • URL: http://$HOST:$PORT"
  echo "  • Environment: ${ENVIRONMENT:-mock}"
  echo "  • Auto-reload: $RELOAD"
  echo
  echo -e "${BLUE}API Documentation:${NC}"
  echo "  • Swagger UI: http://$HOST:$PORT/docs"
  echo "  • ReDoc: http://$HOST:$PORT/redoc"
  echo "  • OpenAPI: http://$HOST:$PORT/openapi.json"
  echo
  echo -e "${BLUE}Health Check:${NC}"
  echo "  • curl http://$HOST:$PORT/api/v1/status/health"
  echo
  echo -e "${BLUE}Logs:${NC}"
  echo "  • tail -f $LOG_FILE"
  echo
  echo -e "${MAGENTA}Press Ctrl+C to stop the server${NC}"
  echo
}

# Setup signal handlers for graceful shutdown
setup_signal_handlers() {
  trap 'on_shutdown' SIGINT SIGTERM
}

on_shutdown() {
  echo
  print_info "Shutting down gracefully..."
  
  if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
  fi
  
  print_ok "Application stopped"
  exit 0
}

# Main entry point
main() {
  parse_args "$@"
  
  print_header
  check_venv
  load_env_dev
  check_dependencies
  create_directories
  check_port
  start_application
  
  # Save PID
  echo $$ > "$PID_FILE"
  
  print_startup_info
  
  # Setup signal handlers
  setup_signal_handlers
  
  # Build uvicorn command
  if [ "$RELOAD" = true ]; then
    UVICORN_CMD="uvicorn src.api.main:app --reload --host $HOST --port $PORT"
  else
    UVICORN_CMD="uvicorn src.api.main:app --host $HOST --port $PORT"
  fi
  
  # Start application (output to both console and log file)
  $UVICORN_CMD 2>&1 | tee -a "$LOG_FILE"
}

main "$@"
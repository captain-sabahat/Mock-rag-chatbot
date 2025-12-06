#!/usr/bin/env bash

################################################################################
# setup_env.sh — Mock Environment Setup (Python ≥3.12, pip ≥25)
#
# Purpose: Create/activate venv, upgrade pip to 25+, install requirements,
#          create .env.dev from template if present, and run optional verification.
#
# Usage:   chmod +x scripts/setup_env.sh && ./scripts/setup_env.sh
#
# This script:
#   1. Checks Python 3.12+ is installed
#   2. Creates virtual environment (venv)
#   3. Activates virtual environment
#   4. Upgrades pip to version 25+
#   5. Installs dependencies from requirements.txt
#   6. Creates .env.dev (from .env.example or minimal template)
#   7. Runs optional verification (verify_installation.py)
#   8. Prints summary and next steps
#
# Environment: Mock (no dev/test/prod profiles)
#
################################################################################

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Helper functions
print_header() {
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║     RAG ADMIN PIPELINE - MOCK SETUP (Python 3.12+ / pip 25+)   ║${NC}"
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

# Step 1: Check Python 3.12+
check_python312() {
  print_info "[1/7] Checking Python ≥ 3.12 ..."
  
  if ! command -v python3 >/dev/null 2>&1; then
    print_error "python3 not found. Please install Python 3.12+"
    exit 1
  fi
  
  MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
  MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
  
  if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 12 ]; }; then
    print_error "Python 3.12+ required (found ${MAJOR}.${MINOR})"
    exit 1
  fi
  
  print_ok "Python ${MAJOR}.${MINOR} detected"
  echo
}

# Step 2: Create virtual environment
create_venv() {
  print_info "[2/7] Creating virtual environment 'venv' ..."
  
  if [ -d "venv" ]; then
    print_info "venv already exists. Skipping creation."
  else
    python3 -m venv venv
    print_ok "Virtual environment created"
  fi
  
  echo
}

# Step 3: Activate virtual environment
activate_venv() {
  print_info "[3/7] Activating virtual environment ..."
  
  # shellcheck disable=SC1091
  source venv/bin/activate
  
  PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
  print_ok "Virtual environment activated (Python: $PYTHON_VERSION)"
  echo
}

# Step 4: Upgrade pip to 25+
upgrade_pip25() {
  print_info "[4/7] Upgrading pip to ≥ 25 ..."
  
  python -m pip install --upgrade --quiet pip
  
  # Verify pip major version >= 25
  PIP_MAJOR=$(python -c 'import pip; print(int(pip.__version__.split(".")[0]))')
  
  if [ "$PIP_MAJOR" -lt 25 ]; then
    print_error "pip 25+ required (found version $PIP_MAJOR.x)"
    exit 1
  fi
  
  PIP_VERSION=$(pip --version | awk '{print $2}')
  print_ok "pip ${PIP_VERSION} upgraded"
  echo
}

# Step 5: Install requirements
install_requirements() {
  print_info "[5/7] Installing dependencies from requirements.txt ..."
  
  if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found"
    exit 1
  fi
  
  pip install --quiet -r requirements.txt
  print_ok "Dependencies installed"
  echo
}

# Step 6: Create .env.dev
create_env_dev() {
  print_info "[6/7] Creating .env.dev (mock environment) ..."
  
  if [ -f ".env.dev" ]; then
    print_ok ".env.dev already exists"
  elif [ -f ".env.example" ]; then
    cp .env.example .env.dev
    print_ok ".env.dev created from .env.example"
    print_info "Edit .env.dev to add any API keys or overrides."
  else
    print_info ".env.example not found. Creating minimal .env.dev"
    cat > .env.dev <<'EOF'
# Mock Environment Configuration
ENVIRONMENT=mock
DEBUG=True
LOG_LEVEL=DEBUG

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=v1

# Database
DATABASE_URL=sqlite:///./data/app.db

# File Processing
MAX_FILE_SIZE=52428800
ALLOWED_FILE_TYPES=.pdf,.txt,.docx,.md,.json,.csv,.html

# Rate Limiting
REQUESTS_PER_MINUTE=100

# Monitoring
MONITORING_ENABLED=True
EOF
    print_ok "Minimal .env.dev created"
    print_info "Edit .env.dev to customize for your environment."
  fi
  
  echo
}

# Step 7: Verification (optional)
verify_installation() {
  print_info "[7/7] Verification ..."
  
  if [ -f "verify_installation.py" ]; then
    if python verify_installation.py; then
      print_ok "Verification passed"
    else
      print_error "Verification script failed"
      exit 1
    fi
  else
    print_info "verify_installation.py not found. Skipping verification."
  fi
  
  echo
}

# Print summary and next steps
print_summary() {
  echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║                   ✅ MOCK SETUP COMPLETE!                       ║${NC}"
  echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo
  echo -e "${BLUE}Environment Details:${NC}"
  echo "  • Python: $(python --version)"
  echo "  • pip: $(pip --version | awk '{print $2}')"
  echo "  • venv: $(pwd)/venv"
  echo "  • Config: .env.dev"
  echo
  echo -e "${BLUE}Next steps:${NC}"
  echo "  1. source venv/bin/activate         # (if not already active)"
  echo "  2. ./scripts/run_backend.sh         # Start the FastAPI server"
  echo "  3. Open: http://localhost:8000/docs # Access Swagger UI"
  echo
  echo -e "${BLUE}To deactivate venv:${NC}  deactivate"
  echo
}

# Main entry point
main() {
  print_header
  check_python312
  create_venv
  activate_venv
  upgrade_pip25
  install_requirements
  create_env_dev
  verify_installation
  print_summary
}

main "$@"
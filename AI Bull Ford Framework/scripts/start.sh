#!/bin/bash

# AI Bull Ford (AIBF) - Startup Script
# This script initializes and starts the AIBF application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AIBF_ENV=${AIBF_ENV:-development}
AIBF_HOST=${AIBF_HOST:-0.0.0.0}
AIBF_PORT=${AIBF_PORT:-8000}
AIBF_WORKERS=${AIBF_WORKERS:-1}
AIBF_LOG_LEVEL=${AIBF_LOG_LEVEL:-INFO}
AIBF_RELOAD=${AIBF_RELOAD:-true}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python version
    if ! python3 --version | grep -E "Python 3\.(10|11|12)" > /dev/null; then
        log_error "Python 3.10+ is required"
        exit 1
    fi
    
    # Check if AIBF is installed
    if ! python3 -c "import aibf" 2>/dev/null; then
        log_warning "AIBF not installed, installing..."
        pip install -e .
    fi
    
    log_success "Dependencies check completed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data logs models config static templates
    
    # Set permissions
    chmod 755 data logs models
    
    # Copy default config if not exists
    if [ ! -f "config/config.yaml" ] && [ -f "config.yaml" ]; then
        cp config.yaml config/config.yaml
        log_info "Copied default configuration"
    fi
    
    log_success "Environment setup completed"
}

check_services() {
    log_info "Checking external services..."
    
    # Check database connection
    if [ -n "$AIBF_DATABASE_URL" ]; then
        log_info "Checking database connection..."
        python3 -c "
import asyncio
from aibf.config import get_config
from aibf.database import check_database_connection

async def check():
    config = get_config()
    if await check_database_connection(config.database.url):
        print('Database connection: OK')
    else:
        print('Database connection: FAILED')
        exit(1)

asyncio.run(check())
" || {
            log_error "Database connection failed"
            exit 1
        }
    fi
    
    # Check Redis connection
    if [ -n "$AIBF_REDIS_URL" ]; then
        log_info "Checking Redis connection..."
        python3 -c "
import redis
import os
from urllib.parse import urlparse

redis_url = os.getenv('AIBF_REDIS_URL', 'redis://localhost:6379/0')
parsed = urlparse(redis_url)
r = redis.Redis(host=parsed.hostname, port=parsed.port, db=parsed.path.lstrip('/') or 0)
try:
    r.ping()
    print('Redis connection: OK')
except:
    print('Redis connection: FAILED')
    exit(1)
" || {
            log_warning "Redis connection failed, continuing without cache"
        }
    fi
    
    log_success "Services check completed"
}

run_migrations() {
    log_info "Running database migrations..."
    
    if command -v aibf-migrate >/dev/null 2>&1; then
        aibf-migrate upgrade head
        log_success "Migrations completed"
    else
        log_warning "Migration tool not found, skipping migrations"
    fi
}

start_server() {
    log_info "Starting AIBF server..."
    log_info "Environment: $AIBF_ENV"
    log_info "Host: $AIBF_HOST"
    log_info "Port: $AIBF_PORT"
    log_info "Workers: $AIBF_WORKERS"
    log_info "Log Level: $AIBF_LOG_LEVEL"
    
    # Build command based on environment
    if [ "$AIBF_ENV" = "development" ]; then
        # Development mode with auto-reload
        exec uvicorn aibf.main:app \
            --host "$AIBF_HOST" \
            --port "$AIBF_PORT" \
            --log-level "$(echo $AIBF_LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
            --reload \
            --reload-dir src \
            --access-log
    elif [ "$AIBF_ENV" = "production" ]; then
        # Production mode with Gunicorn
        if command -v gunicorn >/dev/null 2>&1; then
            exec gunicorn aibf.main:app \
                --bind "$AIBF_HOST:$AIBF_PORT" \
                --workers "$AIBF_WORKERS" \
                --worker-class uvicorn.workers.UvicornWorker \
                --log-level "$(echo $AIBF_LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
                --access-logfile - \
                --error-logfile - \
                --preload
        else
            log_warning "Gunicorn not found, using uvicorn"
            exec uvicorn aibf.main:app \
                --host "$AIBF_HOST" \
                --port "$AIBF_PORT" \
                --workers "$AIBF_WORKERS" \
                --log-level "$(echo $AIBF_LOG_LEVEL | tr '[:upper:]' '[:lower:]')" \
                --access-log
        fi
    else
        # Default mode
        exec aibf-server \
            --host "$AIBF_HOST" \
            --port "$AIBF_PORT" \
            --log-level "$AIBF_LOG_LEVEL"
    fi
}

cleanup() {
    log_info "Shutting down gracefully..."
    # Add any cleanup tasks here
    exit 0
}

# Signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting AI Bull Ford (AIBF) application..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                AIBF_ENV="$2"
                shift 2
                ;;
            --host)
                AIBF_HOST="$2"
                shift 2
                ;;
            --port)
                AIBF_PORT="$2"
                shift 2
                ;;
            --workers)
                AIBF_WORKERS="$2"
                shift 2
                ;;
            --log-level)
                AIBF_LOG_LEVEL="$2"
                shift 2
                ;;
            --no-reload)
                AIBF_RELOAD=false
                shift
                ;;
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --env ENV          Set environment (development|production)"
                echo "  --host HOST        Set host address (default: 0.0.0.0)"
                echo "  --port PORT        Set port number (default: 8000)"
                echo "  --workers NUM      Set number of workers (default: 1)"
                echo "  --log-level LEVEL  Set log level (DEBUG|INFO|WARNING|ERROR)"
                echo "  --no-reload        Disable auto-reload in development"
                echo "  --skip-checks      Skip dependency and service checks"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run startup sequence
    if [ "$SKIP_CHECKS" != "true" ]; then
        check_dependencies
        setup_environment
        check_services
        run_migrations
    fi
    
    # Start the server
    start_server
}

# Run main function
main "$@"
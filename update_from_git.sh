#!/bin/bash

# Crypto Trading System - Server Update Script
# This script updates the server from the latest Git repository

set -e

echo "🔄 Updating Crypto Trading System..."
echo "==============================================="

# Variables
PROJECT_DIR="$HOME/crypto_trading_system"
SERVICE_NAME="crypto-trading"

# Function to log messages
log_info() {
    echo "ℹ️  $1"
}

log_success() {
    echo "✅ $1"
}

log_error() {
    echo "❌ $1"
}

# Check if we're in the correct directory
if [ ! -d "$PROJECT_DIR" ]; then
    log_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# 1. Stop the service
log_info "Stopping service..."
sudo systemctl stop $SERVICE_NAME || log_error "Failed to stop service"

# 2. Kill any remaining processes
log_info "Killing any remaining processes..."
sudo pkill -f "python.*app.py" 2>/dev/null || true
sudo fuser -k 8000/tcp 2>/dev/null || true

# 3. Pull latest changes
log_info "Pulling latest changes from Git..."
git pull origin main

# 4. Update Python dependencies
log_info "Updating Python dependencies..."
cd web_dashboard
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Restart the service
log_info "Starting service..."
sudo systemctl start $SERVICE_NAME

# 6. Wait for service to start
log_info "Waiting for service to start..."
sleep 5

# 7. Check service status
if systemctl is-active --quiet $SERVICE_NAME; then
    log_success "Service is running successfully"
else
    log_error "Service failed to start"
    journalctl -u $SERVICE_NAME -n 10 --no-pager
    exit 1
fi

# 8. Test API endpoint
log_info "Testing API endpoints..."
sleep 2

if curl -s http://localhost:8000 > /dev/null; then
    log_success "Web dashboard is accessible"
else
    log_error "Web dashboard is not accessible"
fi

if curl -s http://localhost:8000/api/binance/symbols > /dev/null; then
    log_success "Binance symbols API is working"
else
    log_error "Binance symbols API is not working"
fi

# 9. Show final status
echo ""
echo "🎉 Update completed successfully!"
echo "==============================================="
echo "🌐 Dashboard: http://34.47.77.230:8000"
echo "🔍 New feature: Dynamic Binance USDT symbol loading"
echo "📊 Backtest: http://34.47.77.230:8000/backtest"
echo ""
echo "🔧 Service status:"
sudo systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo "✨ Features now available:"
echo "• 200+ Binance USDT symbols with search"
echo "• Real-time symbol loading and filtering"
echo "• Fallback to default symbols if API fails"
echo ""
log_success "Ready to use! 🚀"
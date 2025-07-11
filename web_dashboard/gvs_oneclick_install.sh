#!/bin/bash

# 🚀 GVS 서버 크립토 트레이딩 시스템 원클릭 설치
# 완전 자동화된 설치 및 실행 스크립트

set -e

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 아스키 아트 로고
echo -e "${CYAN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🚀 CRYPTO TRADING SYSTEM - GVS SERVER INSTALLER      ║
║                                                           ║
║    ┌─────────────────────────────────────────────────┐   ║
║    │  • Binance USDT Full Symbol Scan               │   ║
║    │  • Real-time Trading Dashboard                 │   ║
║    │  • 24/7 Auto Service                          │   ║
║    │  • Advanced Backtesting                       │   ║
║    │  • Market Phase Analysis                      │   ║
║    └─────────────────────────────────────────────────┘   ║
║                                                           ║
║    Server: 34.47.77.230:5000                            ║
║    Repository: TonyStarK-korean/crypto_trading_system    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 설치 시작 시간 기록
START_TIME=$(date +%s)

# 로그 함수
log_step() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

log_info() {
    echo -e "${PURPLE}[INFO] $1${NC}"
}

# 에러 핸들링
handle_error() {
    log_error "Installation failed at line $1"
    log_error "Please check the error messages above and try again"
    exit 1
}

trap 'handle_error $LINENO' ERR

# 설정 변수
PROJECT_NAME="crypto_trading_system"
REPO_URL="https://github.com/TonyStarK-korean/crypto_trading_system.git"
SERVICE_NAME="crypto-trading"
CURRENT_USER=$(whoami)
HOME_DIR=$(eval echo ~$CURRENT_USER)
PROJECT_DIR="$HOME_DIR/$PROJECT_NAME"
DASHBOARD_DIR="$PROJECT_DIR/web_dashboard"

log_info "Installation Configuration:"
log_info "• User: $CURRENT_USER"
log_info "• Home: $HOME_DIR"
log_info "• Project: $PROJECT_DIR"
log_info "• Service: $SERVICE_NAME"
echo ""

# 1. 시스템 정보 확인
log_step "1/12 System Information Check"
echo "OS: $(uname -s) $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Python: $(python3 --version 2>/dev/null || echo 'Not installed')"
echo "Git: $(git --version 2>/dev/null || echo 'Not installed')"
echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "Available Disk: $(df -h / | tail -1 | awk '{print $4}')"

# 2. 시스템 업데이트 및 필수 패키지 설치
log_step "2/12 System Update & Package Installation"
sudo apt update -qq
sudo apt install -y python3 python3-pip python3-venv git curl wget screen htop ufw build-essential python3-dev

# 3. 기존 프로젝트 및 서비스 정리
log_step "3/12 Cleanup Existing Installation"
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    log_warning "Stopping existing service..."
    sudo systemctl stop $SERVICE_NAME
    sudo systemctl disable $SERVICE_NAME
fi

# 기존 프로세스 종료
sudo pkill -f "python.*app.py" 2>/dev/null || true

# 기존 프로젝트 백업
if [ -d "$PROJECT_DIR" ]; then
    log_warning "Backing up existing project..."
    BACKUP_DIR="${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$PROJECT_DIR" "$BACKUP_DIR"
    log_success "Backup created: $BACKUP_DIR"
fi

# 4. 프로젝트 클론
log_step "4/12 Project Download"
cd "$HOME_DIR"
git clone "$REPO_URL" "$PROJECT_NAME"
cd "$PROJECT_DIR"
log_success "Project cloned successfully"

# 5. 웹 대시보드 설정
log_step "5/12 Web Dashboard Setup"
cd "$DASHBOARD_DIR"

# Python 가상환경 생성
log_info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
log_success "Virtual environment created"

# 6. 패키지 설치
log_step "6/12 Python Package Installation"
pip install --upgrade pip
pip install -r requirements.txt
log_success "All packages installed"

# 7. 필요한 디렉토리 생성
log_step "7/12 Directory Structure Setup"
mkdir -p data logs backup
chmod 755 data logs backup
log_success "Directory structure created"

# 8. 방화벽 설정
log_step "8/12 Firewall Configuration"
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
log_success "Firewall configured"

# 9. systemd 서비스 생성
log_step "9/12 System Service Creation"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$DASHBOARD_DIR
Environment=PATH=$DASHBOARD_DIR/venv/bin
Environment=PYTHONPATH=$DASHBOARD_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$DASHBOARD_DIR/venv/bin/python app.py
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

log_success "Service file created"

# 10. 서비스 등록 및 시작
log_step "10/12 Service Registration & Startup"
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME
log_success "Service started"

# 11. 관리 스크립트 생성
log_step "11/12 Management Scripts Creation"

# 서비스 관리 스크립트
cat > "$DASHBOARD_DIR/manage_service.sh" << 'EOF'
#!/bin/bash
SERVICE_NAME="crypto-trading"
case "$1" in
    start)   sudo systemctl start $SERVICE_NAME ;;
    stop)    sudo systemctl stop $SERVICE_NAME ;;
    restart) sudo systemctl restart $SERVICE_NAME ;;
    status)  sudo systemctl status $SERVICE_NAME ;;
    logs)    journalctl -u $SERVICE_NAME -f ;;
    *)       echo "Usage: $0 {start|stop|restart|status|logs}" ;;
esac
EOF

# 빠른 시작 스크립트
cat > "$DASHBOARD_DIR/quick_start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
EOF

# 상태 확인 스크립트
cat > "$DASHBOARD_DIR/check_status.sh" << 'EOF'
#!/bin/bash
echo "🔍 Crypto Trading System Status Check"
echo "======================================"
echo "Service Status:"
sudo systemctl status crypto-trading --no-pager -l
echo ""
echo "Network Status:"
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ Web Dashboard: ONLINE"
else
    echo "❌ Web Dashboard: OFFLINE"
fi
echo ""
echo "System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%"), $3/$2 * 100.0}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
echo ""
echo "Recent Logs:"
journalctl -u crypto-trading -n 5 --no-pager
EOF

chmod +x "$DASHBOARD_DIR"/*.sh
log_success "Management scripts created"

# 12. 설치 완료 확인
log_step "12/12 Installation Verification"
sleep 5

# 서비스 상태 확인
if systemctl is-active --quiet $SERVICE_NAME; then
    log_success "Service is running"
else
    log_error "Service failed to start"
    journalctl -u $SERVICE_NAME -n 10 --no-pager
fi

# 웹 접속 테스트
if curl -s http://localhost:5000 > /dev/null; then
    log_success "Web dashboard is accessible"
else
    log_warning "Web dashboard might not be ready yet"
fi

# 설치 완료 시간 계산
END_TIME=$(date +%s)
INSTALLATION_TIME=$((END_TIME - START_TIME))

# 완료 메시지
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║                 🎉 INSTALLATION COMPLETED! 🎉             ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Installation Summary:${NC}"
echo "• Installation time: ${INSTALLATION_TIME} seconds"
echo "• Project directory: $PROJECT_DIR"
echo "• Service name: $SERVICE_NAME"
echo "• Current user: $CURRENT_USER"
echo ""
echo -e "${CYAN}🌐 Access URLs:${NC}"
echo "• External: http://34.47.77.230:5000"
echo "• Local: http://localhost:5000"
echo "• Backtest: http://34.47.77.230:5000/backtest"
echo "• Live Trading: http://34.47.77.230:5000/live_trading"
echo ""
echo -e "${YELLOW}🔧 Management Commands:${NC}"
echo "• Service status: sudo systemctl status $SERVICE_NAME"
echo "• View logs: journalctl -u $SERVICE_NAME -f"
echo "• Restart service: sudo systemctl restart $SERVICE_NAME"
echo ""
echo -e "${PURPLE}⚡ Quick Scripts:${NC}"
echo "• Check status: ./check_status.sh"
echo "• Manage service: ./manage_service.sh [start|stop|restart|status|logs]"
echo "• Quick start: ./quick_start.sh"
echo ""
echo -e "${GREEN}🎯 Features Available:${NC}"
echo "• ✅ Binance USDT Full Symbol Scan (200+ symbols)"
echo "• ✅ Real-time Trading Dashboard"
echo "• ✅ Advanced Backtesting with Market Phase Analysis"
echo "• ✅ 24/7 Auto Service with Auto-restart"
echo "• ✅ Detailed Trade Records & Statistics"
echo "• ✅ Mobile-friendly Responsive Design"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "1. Open your browser and go to: http://34.47.77.230:5000"
echo "2. Test the backtesting system with full symbol scan"
echo "3. Configure real-time trading settings"
echo "4. Monitor system performance and logs"
echo ""
echo -e "${GREEN}🎉 Your Crypto Trading System is now running 24/7! 🎉${NC}"
echo ""
echo -e "${CYAN}Happy Trading! 📈💰🚀${NC}"
echo ""

# 최종 상태 출력
echo -e "${BLUE}Current Status:${NC}"
sudo systemctl status $SERVICE_NAME --no-pager -l || true
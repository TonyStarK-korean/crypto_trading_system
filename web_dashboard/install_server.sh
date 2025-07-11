#!/bin/bash

# 🚀 Crypto Trading System - One-Click Server Installation
# GVS Server 34.47.77.230 원클릭 설치 스크립트

set -e

echo "🚀 Crypto Trading System - Server Installation"
echo "=================================================="
echo "Server: 34.47.77.230:5000"
echo "Repository: https://github.com/TonyStarK-korean/crypto_trading_system"
echo "Time: $(date)"
echo ""

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
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

# 에러 핸들링
handle_error() {
    log_error "Installation failed at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# 환경 설정
PROJECT_DIR="/opt/crypto_trading_system"
SERVICE_NAME="crypto-trading"
REPO_URL="https://github.com/TonyStarK-korean/crypto_trading_system.git"
USER=$(whoami)

log_info "Starting installation process..."

# 1. 시스템 업데이트
log_info "1/10 Updating system packages..."
sudo apt update && sudo apt upgrade -y
log_success "System updated"

# 2. 필수 패키지 설치
log_info "2/10 Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv git nginx ufw curl wget htop
log_success "Required packages installed"

# 3. 방화벽 설정
log_info "3/10 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
log_success "Firewall configured"

# 4. 프로젝트 디렉토리 설정
log_info "4/10 Setting up project directory..."
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# 기존 프로젝트가 있다면 백업
if [ -d "$PROJECT_DIR/.git" ]; then
    log_warning "Existing project found, creating backup..."
    sudo cp -r $PROJECT_DIR ${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)
    cd $PROJECT_DIR
    git pull origin main
    log_success "Project updated"
else
    log_info "Cloning repository..."
    git clone $REPO_URL $PROJECT_DIR
    cd $PROJECT_DIR
    log_success "Repository cloned"
fi

# 5. 웹 대시보드 설정
log_info "5/10 Setting up web dashboard..."
cd $PROJECT_DIR/web_dashboard

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 필요한 디렉토리 생성
mkdir -p data logs backup

log_success "Web dashboard setup completed"

# 6. 기존 서비스 중지
log_info "6/10 Stopping existing services..."
sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
sudo pkill -f "python.*app.py" 2>/dev/null || true
log_success "Existing services stopped"

# 7. Systemd 서비스 생성
log_info "7/10 Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR/web_dashboard
Environment=PATH=$PROJECT_DIR/web_dashboard/venv/bin
Environment=PYTHONPATH=$PROJECT_DIR/web_dashboard
ExecStart=$PROJECT_DIR/web_dashboard/venv/bin/python app.py
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading

[Install]
WantedBy=multi-user.target
EOF

log_success "Systemd service created"

# 8. 서비스 활성화 및 시작
log_info "8/10 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

# 서비스 상태 확인
sleep 5
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    log_success "Service started successfully"
else
    log_error "Service failed to start"
    sudo systemctl status $SERVICE_NAME
fi

# 9. Nginx 설정
log_info "9/10 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/$SERVICE_NAME > /dev/null <<EOF
server {
    listen 80;
    server_name 34.47.77.230;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    location /static {
        alias $PROJECT_DIR/web_dashboard/static;
        expires 30d;
    }
}
EOF

# Nginx 설정 활성화
sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

log_success "Nginx configured"

# 10. 자동 업데이트 설정
log_info "10/10 Setting up auto-update..."
cd $PROJECT_DIR/web_dashboard

# 자동 업데이트 서비스 생성
sudo tee /etc/systemd/system/crypto-auto-update.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Auto Update
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR/web_dashboard
Environment=PATH=$PROJECT_DIR/web_dashboard/venv/bin
ExecStart=$PROJECT_DIR/web_dashboard/venv/bin/python auto_update.py --daemon
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crypto-auto-update
sudo systemctl start crypto-auto-update

log_success "Auto-update service configured"

# 최종 상태 확인
log_info "Checking final status..."
sleep 5

echo ""
echo "🎉 Installation Completed Successfully!"
echo "=================================================="
echo "🌐 Web Dashboard URLs:"
echo "  • Direct: http://34.47.77.230:5000"
echo "  • Nginx:  http://34.47.77.230"
echo ""
echo "📊 Service Status:"
sudo systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo "🔧 Service Management Commands:"
echo "  • Status:  sudo systemctl status $SERVICE_NAME"
echo "  • Start:   sudo systemctl start $SERVICE_NAME"
echo "  • Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  • Restart: sudo systemctl restart $SERVICE_NAME"
echo "  • Logs:    journalctl -u $SERVICE_NAME -f"
echo ""
echo "🔄 Auto-Update:"
echo "  • Status:  sudo systemctl status crypto-auto-update"
echo "  • Logs:    journalctl -u crypto-auto-update -f"
echo ""
echo "🛠️ Troubleshooting:"
echo "  • Check logs: tail -f $PROJECT_DIR/web_dashboard/logs/app.log"
echo "  • Test connectivity: curl http://localhost:5000"
echo "  • Restart all: sudo systemctl restart $SERVICE_NAME nginx"
echo ""
echo "📈 Performance Monitoring:"
echo "  • System resources: htop"
echo "  • Disk usage: df -h"
echo "  • Network: netstat -tulpn | grep :5000"
echo ""
echo "🔐 Security:"
echo "  • Firewall status: sudo ufw status"
echo "  • Service logs: sudo journalctl -u $SERVICE_NAME"
echo ""
echo "✅ Crypto Trading System is now running 24/7!"
echo "🚀 Happy Trading!"

# 접속 테스트
log_info "Testing web dashboard connectivity..."
if curl -s http://localhost:5000 > /dev/null; then
    log_success "Web dashboard is accessible!"
else
    log_warning "Web dashboard might not be fully ready yet. Please check logs."
fi

echo ""
echo "📋 Next Steps:"
echo "1. Open your browser and go to: http://34.47.77.230:5000"
echo "2. Test the backtesting system with full symbol scan"
echo "3. Configure real-time trading settings"
echo "4. Monitor system performance and logs"
echo ""
echo "🎯 Installation Complete! 🎯"
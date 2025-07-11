#!/bin/bash

# Crypto Trading System GVS Server Deployment Script
# GVS 서버 자동 배포 스크립트

set -e

echo "🚀 Starting Crypto Trading System Deployment..."
echo "Server: 34.47.77.230:5000"
echo "Time: $(date)"

# 환경 설정
PROJECT_DIR="/opt/crypto_trading_system"
SERVICE_NAME="crypto-trading"
REPO_URL="https://github.com/TonyStarK-korean/crypto_trading_system.git"

# 시스템 업데이트
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
echo "🔧 Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv git nginx ufw

# 방화벽 설정
echo "🔒 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 프로젝트 디렉토리 생성
echo "📁 Setting up project directory..."
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Git 저장소 클론 또는 업데이트
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "🔄 Updating existing repository..."
    cd $PROJECT_DIR
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone $REPO_URL $PROJECT_DIR
    cd $PROJECT_DIR
fi

# 웹 대시보드 디렉토리로 이동
cd $PROJECT_DIR/web_dashboard

# Python 가상환경 생성
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 데이터 디렉토리 생성
echo "📂 Creating data directory..."
mkdir -p data

# 기존 서비스 중지
echo "🛑 Stopping existing services..."
sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
sudo pkill -f "python.*app.py" 2>/dev/null || true

# Systemd 서비스 파일 생성
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/web_dashboard
Environment=PATH=$PROJECT_DIR/web_dashboard/venv/bin
ExecStart=$PROJECT_DIR/web_dashboard/venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화 및 시작
echo "🔄 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

# Nginx 설정
echo "🌐 Configuring Nginx reverse proxy..."
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
    }
}
EOF

# Nginx 설정 활성화
sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# 서비스 상태 확인
echo "📊 Checking service status..."
sudo systemctl status $SERVICE_NAME --no-pager

# 완료 메시지
echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Dashboard URLs:"
echo "  • Direct: http://34.47.77.230:5000"
echo "  • Nginx: http://34.47.77.230"
echo ""
echo "🔧 Service management commands:"
echo "  • Status: sudo systemctl status $SERVICE_NAME"
echo "  • Start: sudo systemctl start $SERVICE_NAME"
echo "  • Stop: sudo systemctl stop $SERVICE_NAME"
echo "  • Restart: sudo systemctl restart $SERVICE_NAME"
echo "  • Logs: journalctl -u $SERVICE_NAME -f"
echo ""
echo "🔄 To update the system:"
echo "  cd $PROJECT_DIR && git pull && sudo systemctl restart $SERVICE_NAME"
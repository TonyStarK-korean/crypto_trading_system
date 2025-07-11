#!/bin/bash

# 🚀 크립토 트레이딩 시스템 24시간 자동 가동 설정 스크립트

set -e

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 크립토 트레이딩 시스템 24시간 자동 가동 설정${NC}"
echo "================================================="

# 설정 변수
SERVICE_NAME="crypto-trading"
PROJECT_DIR="$HOME/crypto_trading_system"
DASHBOARD_DIR="$PROJECT_DIR/web_dashboard"
USER_NAME=$(whoami)

# 로그 함수
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. 기존 서비스 중지 및 제거
log_info "기존 서비스 정리..."
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    sudo systemctl stop $SERVICE_NAME
    log_success "기존 서비스 중지됨"
fi

if systemctl is-enabled --quiet $SERVICE_NAME 2>/dev/null; then
    sudo systemctl disable $SERVICE_NAME
    log_success "기존 서비스 비활성화됨"
fi

# 2. 기존 프로세스 종료
log_info "기존 프로세스 종료..."
sudo pkill -f "python.*app.py" 2>/dev/null || true
sudo fuser -k 8000/tcp 2>/dev/null || true
log_success "기존 프로세스 종료됨"

# 3. 디렉토리 확인
if [ ! -d "$DASHBOARD_DIR" ]; then
    log_error "프로젝트 디렉토리를 찾을 수 없습니다: $DASHBOARD_DIR"
    exit 1
fi

# 4. 가상환경 확인
if [ ! -d "$DASHBOARD_DIR/venv" ]; then
    log_info "가상환경 생성 중..."
    cd "$DASHBOARD_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "가상환경 생성 완료"
fi

# 5. systemd 서비스 파일 생성
log_info "systemd 서비스 파일 생성..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard - 24/7 Auto Service
Documentation=https://github.com/TonyStarK-korean/crypto_trading_system
After=network.target network-online.target
Wants=network-online.target
RequiresMountsFor=/home

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$DASHBOARD_DIR
Environment=PATH=$DASHBOARD_DIR/venv/bin
Environment=PYTHONPATH=$DASHBOARD_DIR
Environment=PYTHONUNBUFFERED=1
Environment=FLASK_APP=app.py
Environment=FLASK_ENV=production
ExecStart=$DASHBOARD_DIR/venv/bin/python app.py
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30

# 로그 관리
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading

# 보안 설정
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# 리소스 제한
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

log_success "systemd 서비스 파일 생성됨"

# 6. 서비스 등록 및 활성화
log_info "서비스 등록 및 활성화..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
log_success "서비스 활성화됨 (부팅 시 자동 시작)"

# 7. 서비스 시작
log_info "서비스 시작..."
sudo systemctl start $SERVICE_NAME
log_success "서비스 시작됨"

# 8. 서비스 상태 확인
log_info "서비스 상태 확인 중..."
sleep 5

if systemctl is-active --quiet $SERVICE_NAME; then
    log_success "서비스가 정상적으로 실행 중입니다"
else
    log_error "서비스 시작에 실패했습니다"
    echo "에러 로그:"
    journalctl -u $SERVICE_NAME -n 10 --no-pager
    exit 1
fi

# 9. 웹 접속 테스트
log_info "웹 접속 테스트..."
sleep 3

if curl -s http://localhost:8000 > /dev/null; then
    log_success "웹 대시보드 접속 가능"
else
    log_warning "웹 대시보드 접속 테스트 실패 (잠시 후 다시 시도해보세요)"
fi

# 10. 방화벽 설정
log_info "방화벽 설정 확인..."
sudo ufw allow 8000/tcp
sudo ufw allow 22/tcp
if ! sudo ufw status | grep -q "Status: active"; then
    sudo ufw --force enable
fi
log_success "방화벽 설정 완료"

# 11. 서비스 관리 명령어 안내
echo ""
echo -e "${GREEN}🎉 24시간 자동 가동 설정 완료! 🎉${NC}"
echo "================================================="
echo ""
echo -e "${BLUE}📋 서비스 관리 명령어:${NC}"
echo "• 서비스 상태 확인: sudo systemctl status $SERVICE_NAME"
echo "• 서비스 중지: sudo systemctl stop $SERVICE_NAME"
echo "• 서비스 시작: sudo systemctl start $SERVICE_NAME"
echo "• 서비스 재시작: sudo systemctl restart $SERVICE_NAME"
echo "• 로그 확인: journalctl -u $SERVICE_NAME -f"
echo "• 최근 로그: journalctl -u $SERVICE_NAME -n 50"
echo ""
echo -e "${BLUE}🌐 접속 정보:${NC}"
echo "• 웹 대시보드: http://34.47.77.230:8000"
echo "• 백테스트: http://34.47.77.230:8000/backtest"
echo "• 실시간 거래: http://34.47.77.230:8000/live_trading"
echo ""
echo -e "${BLUE}✨ 특징:${NC}"
echo "• ✅ SSH 터미널 종료 후에도 24시간 가동"
echo "• ✅ 서버 재부팅 시 자동 시작"
echo "• ✅ 프로세스 종료 시 자동 재시작"
echo "• ✅ 로그 자동 관리 (journalctl)"
echo "• ✅ 보안 설정 적용"
echo "• ✅ 리소스 제한 설정"
echo ""
echo -e "${GREEN}🚀 이제 SSH를 종료해도 웹서버가 계속 실행됩니다! 🚀${NC}"
echo ""

# 12. 현재 상태 표시
echo -e "${BLUE}📊 현재 상태:${NC}"
sudo systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo -e "${BLUE}🔍 프로세스 확인:${NC}"
ps aux | grep -E "(python.*app\.py|crypto)" | grep -v grep || echo "프로세스 실행 중..."
echo ""
echo -e "${BLUE}🌐 네트워크 상태:${NC}"
netstat -tlnp | grep :8000 || echo "포트 8000 대기 중..."
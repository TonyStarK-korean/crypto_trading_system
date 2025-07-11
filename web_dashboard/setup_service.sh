#!/bin/bash

# 🚀 GVS 서버 24시간 서비스 설정 스크립트

set -e

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 GVS 서버 24시간 서비스 설정${NC}"
echo "=================================================="

# 현재 사용자 및 경로 확인
CURRENT_USER=$(whoami)
PROJECT_PATH=$(pwd)
SERVICE_NAME="crypto-trading"

echo "사용자: $CURRENT_USER"
echo "프로젝트 경로: $PROJECT_PATH"
echo "서비스 이름: $SERVICE_NAME"

# 1. 기존 서비스 중지
echo -e "${BLUE}1. 기존 서비스 중지${NC}"
sudo systemctl stop $SERVICE_NAME 2>/dev/null || echo "기존 서비스 없음"
sudo pkill -f "python.*app.py" 2>/dev/null || echo "실행 중인 앱 없음"

# 2. systemd 서비스 파일 생성
echo -e "${BLUE}2. systemd 서비스 파일 생성${NC}"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$PROJECT_PATH
Environment=PATH=$PROJECT_PATH/venv/bin
Environment=PYTHONPATH=$PROJECT_PATH
ExecStart=$PROJECT_PATH/venv/bin/python app.py
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}서비스 파일 생성 완료${NC}"

# 3. 서비스 등록 및 시작
echo -e "${BLUE}3. 서비스 등록 및 시작${NC}"
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

# 4. 서비스 상태 확인
echo -e "${BLUE}4. 서비스 상태 확인${NC}"
sleep 3
sudo systemctl status $SERVICE_NAME --no-pager

# 5. 방화벽 설정
echo -e "${BLUE}5. 방화벽 설정${NC}"
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw --force enable
echo -e "${GREEN}방화벽 설정 완료${NC}"

# 6. 자동 시작 스크립트 생성
echo -e "${BLUE}6. 관리 스크립트 생성${NC}"
cat > manage_service.sh << 'EOF'
#!/bin/bash
# 서비스 관리 스크립트

SERVICE_NAME="crypto-trading"

case "$1" in
    start)
        echo "서비스 시작 중..."
        sudo systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "서비스 중지 중..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "서비스 재시작 중..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    status)
        sudo systemctl status $SERVICE_NAME
        ;;
    logs)
        journalctl -u $SERVICE_NAME -f
        ;;
    *)
        echo "사용법: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF

chmod +x manage_service.sh
echo -e "${GREEN}관리 스크립트 생성 완료: ./manage_service.sh${NC}"

# 7. 연결 테스트
echo -e "${BLUE}7. 연결 테스트${NC}"
sleep 5
if curl -s http://localhost:5000 > /dev/null; then
    echo -e "${GREEN}✅ 웹 대시보드 접속 성공!${NC}"
else
    echo -e "${RED}❌ 웹 대시보드 접속 실패${NC}"
    echo "로그 확인: journalctl -u $SERVICE_NAME -f"
fi

# 8. 완료 메시지
echo ""
echo "=================================================="
echo -e "${GREEN}🎉 24시간 서비스 설정 완료!${NC}"
echo "=================================================="
echo -e "${BLUE}📊 접속 URL:${NC}"
echo "• 외부 접속: http://34.47.77.230:5000"
echo "• 로컬 테스트: http://localhost:5000"
echo ""
echo -e "${BLUE}🔧 서비스 관리:${NC}"
echo "• 상태 확인: sudo systemctl status $SERVICE_NAME"
echo "• 시작: sudo systemctl start $SERVICE_NAME"
echo "• 중지: sudo systemctl stop $SERVICE_NAME"
echo "• 재시작: sudo systemctl restart $SERVICE_NAME"
echo "• 로그: journalctl -u $SERVICE_NAME -f"
echo ""
echo -e "${BLUE}⚡ 빠른 명령어:${NC}"
echo "• ./manage_service.sh start"
echo "• ./manage_service.sh stop"
echo "• ./manage_service.sh restart"
echo "• ./manage_service.sh status"
echo "• ./manage_service.sh logs"
echo ""
echo -e "${YELLOW}💡 팁:${NC}"
echo "• 서비스는 시스템 부팅 시 자동으로 시작됩니다"
echo "• 서비스가 중지되면 자동으로 재시작됩니다"
echo "• 로그는 systemd journal에 저장됩니다"
echo ""
echo -e "${GREEN}🚀 웹 대시보드가 24시간 가동 중입니다!${NC}"
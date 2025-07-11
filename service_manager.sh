#!/bin/bash

# 🔧 크립토 트레이딩 시스템 서비스 관리자
# 24시간 자동 가동 서비스 관리를 위한 통합 스크립트

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SERVICE_NAME="crypto-trading"

# 로고 출력
show_logo() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🚀 CRYPTO TRADING SYSTEM - SERVICE MANAGER               ║
║                                                              ║
║    ┌────────────────────────────────────────────────────┐   ║
║    │  24시간 자동 가동 서비스 관리 도구                │   ║
║    │  SSH 터미널 종료 후에도 계속 실행                 │   ║
║    └────────────────────────────────────────────────────┘   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 서비스 상태 확인
check_status() {
    echo -e "${BLUE}🔍 서비스 상태 확인${NC}"
    echo "=============================="
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ 서비스 상태: 실행 중${NC}"
    else
        echo -e "${RED}❌ 서비스 상태: 중지됨${NC}"
    fi
    
    if systemctl is-enabled --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ 자동 시작: 활성화됨${NC}"
    else
        echo -e "${RED}❌ 자동 시작: 비활성화됨${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}📊 상세 상태:${NC}"
    sudo systemctl status $SERVICE_NAME --no-pager -l
    
    echo ""
    echo -e "${BLUE}🌐 네트워크 상태:${NC}"
    if curl -s http://localhost:8000 > /dev/null; then
        echo -e "${GREEN}✅ 웹 대시보드: 접속 가능${NC}"
    else
        echo -e "${RED}❌ 웹 대시보드: 접속 불가${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}🔍 프로세스 상태:${NC}"
    ps aux | grep -E "(python.*app\.py)" | grep -v grep | head -5
    
    echo ""
    echo -e "${BLUE}📡 포트 상태:${NC}"
    netstat -tlnp | grep :8000 || echo "포트 8000: 사용 중이지 않음"
}

# 서비스 시작
start_service() {
    echo -e "${BLUE}🚀 서비스 시작${NC}"
    echo "================="
    
    sudo systemctl start $SERVICE_NAME
    sleep 3
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ 서비스가 성공적으로 시작되었습니다${NC}"
    else
        echo -e "${RED}❌ 서비스 시작에 실패했습니다${NC}"
        echo "에러 로그:"
        journalctl -u $SERVICE_NAME -n 10 --no-pager
    fi
}

# 서비스 중지
stop_service() {
    echo -e "${BLUE}🛑 서비스 중지${NC}"
    echo "================="
    
    sudo systemctl stop $SERVICE_NAME
    sleep 2
    
    if ! systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ 서비스가 성공적으로 중지되었습니다${NC}"
    else
        echo -e "${RED}❌ 서비스 중지에 실패했습니다${NC}"
    fi
}

# 서비스 재시작
restart_service() {
    echo -e "${BLUE}🔄 서비스 재시작${NC}"
    echo "==================="
    
    sudo systemctl restart $SERVICE_NAME
    sleep 5
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ 서비스가 성공적으로 재시작되었습니다${NC}"
    else
        echo -e "${RED}❌ 서비스 재시작에 실패했습니다${NC}"
        echo "에러 로그:"
        journalctl -u $SERVICE_NAME -n 10 --no-pager
    fi
}

# 로그 보기
show_logs() {
    echo -e "${BLUE}📋 실시간 로그 보기${NC}"
    echo "====================="
    echo "종료하려면 Ctrl+C를 누르세요"
    echo ""
    
    journalctl -u $SERVICE_NAME -f
}

# 최근 로그 보기
show_recent_logs() {
    echo -e "${BLUE}📋 최근 로그 (50줄)${NC}"
    echo "====================="
    
    journalctl -u $SERVICE_NAME -n 50 --no-pager
}

# 서비스 활성화
enable_service() {
    echo -e "${BLUE}✅ 서비스 자동 시작 활성화${NC}"
    echo "============================="
    
    sudo systemctl enable $SERVICE_NAME
    echo -e "${GREEN}✅ 서버 재부팅 시 자동으로 시작됩니다${NC}"
}

# 서비스 비활성화
disable_service() {
    echo -e "${BLUE}❌ 서비스 자동 시작 비활성화${NC}"
    echo "==============================="
    
    sudo systemctl disable $SERVICE_NAME
    echo -e "${YELLOW}⚠️  서버 재부팅 시 자동으로 시작되지 않습니다${NC}"
}

# 전체 시스템 점검
full_check() {
    echo -e "${BLUE}🔧 전체 시스템 점검${NC}"
    echo "===================="
    
    check_status
    
    echo ""
    echo -e "${BLUE}💾 디스크 사용량:${NC}"
    df -h /
    
    echo ""
    echo -e "${BLUE}🧠 메모리 사용량:${NC}"
    free -h
    
    echo ""
    echo -e "${BLUE}⚡ CPU 사용량:${NC}"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}'
    
    echo ""
    echo -e "${BLUE}🔥 최근 시스템 에러:${NC}"
    journalctl -p err -n 5 --no-pager
}

# 도움말 메시지
show_help() {
    echo -e "${BLUE}📖 사용법:${NC}"
    echo "=========="
    echo "  $0 [명령어]"
    echo ""
    echo -e "${BLUE}사용 가능한 명령어:${NC}"
    echo "  status    - 서비스 상태 확인"
    echo "  start     - 서비스 시작"
    echo "  stop      - 서비스 중지"
    echo "  restart   - 서비스 재시작"
    echo "  logs      - 실시간 로그 보기"
    echo "  recent    - 최근 로그 보기"
    echo "  enable    - 자동 시작 활성화"
    echo "  disable   - 자동 시작 비활성화"
    echo "  check     - 전체 시스템 점검"
    echo "  help      - 도움말 보기"
    echo ""
    echo -e "${BLUE}예시:${NC}"
    echo "  $0 status   # 서비스 상태 확인"
    echo "  $0 restart  # 서비스 재시작"
    echo "  $0 logs     # 실시간 로그 보기"
}

# 메인 로직
main() {
    show_logo
    
    case "$1" in
        status)   check_status ;;
        start)    start_service ;;
        stop)     stop_service ;;
        restart)  restart_service ;;
        logs)     show_logs ;;
        recent)   show_recent_logs ;;
        enable)   enable_service ;;
        disable)  disable_service ;;
        check)    full_check ;;
        help)     show_help ;;
        *)        
            echo -e "${RED}❌ 올바르지 않은 명령어입니다${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
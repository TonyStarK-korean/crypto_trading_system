#!/bin/bash

# 🚀 GVS 서버 웹 대시보드 원클릭 실행 스크립트

set -e

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 GVS 서버 웹 대시보드 시작${NC}"
echo "=================================================="

# 1. 시스템 정보 확인
echo -e "${BLUE}1. 시스템 정보 확인${NC}"
echo "OS: $(uname -s)"
echo "Python: $(python3 --version)"
echo "Current User: $(whoami)"
echo "Current Directory: $(pwd)"

# 2. 필요한 패키지 설치
echo -e "${BLUE}2. 필요한 패키지 설치${NC}"
if ! command -v git &> /dev/null; then
    echo "Git 설치 중..."
    sudo apt update
    sudo apt install -y git python3 python3-pip python3-venv
fi

# 3. 프로젝트 다운로드
echo -e "${BLUE}3. 프로젝트 다운로드${NC}"
if [ ! -d "crypto_trading_system" ]; then
    echo "저장소 클론 중..."
    git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
else
    echo "기존 프로젝트 업데이트 중..."
    cd crypto_trading_system
    git pull origin main
    cd ..
fi

# 4. 웹 대시보드 디렉토리로 이동
echo -e "${BLUE}4. 웹 대시보드 설정${NC}"
cd crypto_trading_system/web_dashboard

# 5. Python 가상환경 설정
echo -e "${BLUE}5. Python 환경 설정${NC}"
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate
echo "가상환경 활성화 완료"

# 6. 패키지 설치
echo -e "${BLUE}6. 패키지 설치${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 7. 필요한 디렉토리 생성
echo -e "${BLUE}7. 디렉토리 생성${NC}"
mkdir -p data logs

# 8. 방화벽 설정
echo -e "${BLUE}8. 방화벽 설정${NC}"
sudo ufw allow 5000/tcp 2>/dev/null || echo "방화벽 설정 건너뛰기"

# 9. 포트 확인
echo -e "${BLUE}9. 포트 상태 확인${NC}"
if lsof -i :5000 &> /dev/null; then
    echo -e "${YELLOW}포트 5000이 사용 중입니다. 기존 프로세스를 종료합니다.${NC}"
    sudo pkill -f "python.*app.py" 2>/dev/null || true
    sleep 2
fi

# 10. 웹 대시보드 시작
echo -e "${GREEN}10. 웹 대시보드 시작${NC}"
echo "=================================================="
echo -e "${GREEN}🌐 웹 대시보드 시작 중...${NC}"
echo "접속 URL: http://34.47.77.230:5000"
echo "로컬 테스트: http://localhost:5000"
echo ""
echo -e "${YELLOW}서버를 중지하려면 Ctrl+C를 누르세요${NC}"
echo "=================================================="

# 서버 시작
python app.py
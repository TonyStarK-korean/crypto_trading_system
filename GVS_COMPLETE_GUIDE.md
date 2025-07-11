# 🚀 GVS 서버 완전 실행 가이드

## 📋 단계별 실행 방법

### 1단계: SSH 접속
```bash
# SSH로 GVS 서버에 접속
ssh ubuntu@34.47.77.230
```

### 2단계: 원클릭 설치 및 실행
```bash
# 홈 디렉토리에서 실행
cd ~

# 프로젝트 다운로드 및 설정
curl -sSL https://raw.githubusercontent.com/TonyStarK-korean/crypto_trading_system/main/web_dashboard/start_gvs.sh | bash
```

### 3단계: 24시간 서비스 설정 (선택사항)
```bash
# 웹 대시보드 디렉토리로 이동
cd ~/crypto_trading_system/web_dashboard

# 24시간 서비스 설정
chmod +x setup_service.sh
./setup_service.sh
```

## 🎯 빠른 실행 (단축 버전)

### 방법 1: 임시 실행 (테스트용)
```bash
# SSH 접속 후 한 번에 실행
ssh ubuntu@34.47.77.230
cd ~
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
cd crypto_trading_system/web_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 방법 2: 백그라운드 실행
```bash
# SSH 접속 후 실행
ssh ubuntu@34.47.77.230
cd ~
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
cd crypto_trading_system/web_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup python app.py > app.log 2>&1 &
```

### 방법 3: Screen 사용 (권장)
```bash
# SSH 접속 후 실행
ssh ubuntu@34.47.77.230
sudo apt install screen -y
screen -S crypto_trading
cd ~
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
cd crypto_trading_system/web_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

# Screen에서 빠져나오기: Ctrl+A, D
# 다시 연결: screen -r crypto_trading
```

## 🌐 접속 확인

### 웹 브라우저에서 접속
- **메인 접속**: http://34.47.77.230:5000
- **백테스트**: http://34.47.77.230:5000/backtest
- **실시간 거래**: http://34.47.77.230:5000/live_trading

### 서버에서 테스트
```bash
# 로컬 접속 테스트
curl http://localhost:5000

# 외부 접속 테스트
curl http://34.47.77.230:5000
```

## 🔧 서비스 관리

### 서비스 설정 후 관리 명령어
```bash
# 서비스 상태 확인
sudo systemctl status crypto-trading

# 서비스 시작/중지/재시작
sudo systemctl start crypto-trading
sudo systemctl stop crypto-trading
sudo systemctl restart crypto-trading

# 로그 확인
journalctl -u crypto-trading -f
```

### 관리 스크립트 사용
```bash
# 웹 대시보드 디렉토리에서
./manage_service.sh start     # 시작
./manage_service.sh stop      # 중지
./manage_service.sh restart   # 재시작
./manage_service.sh status    # 상태 확인
./manage_service.sh logs      # 로그 확인
```

## 📊 문제 해결

### 1. 접속이 안 될 때
```bash
# 포트 확인
sudo lsof -i :5000

# 방화벽 확인
sudo ufw status

# 방화벽 허용
sudo ufw allow 5000/tcp
```

### 2. 서비스가 시작되지 않을 때
```bash
# 로그 확인
journalctl -u crypto-trading -n 50

# 수동 실행으로 에러 확인
cd ~/crypto_trading_system/web_dashboard
source venv/bin/activate
python app.py
```

### 3. 패키지 설치 에러
```bash
# 시스템 업데이트
sudo apt update
sudo apt install python3-dev build-essential

# 가상환경 재생성
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🎉 완료 체크리스트

### 기본 실행 확인
- [ ] SSH 접속 성공
- [ ] 프로젝트 다운로드 완료
- [ ] 가상환경 생성 및 활성화
- [ ] 패키지 설치 완료
- [ ] 웹 대시보드 실행 성공

### 웹 접속 확인
- [ ] http://34.47.77.230:5000 접속 성공
- [ ] 메인 페이지 로드 확인
- [ ] 백테스트 페이지 접속 확인
- [ ] 실시간 거래 페이지 접속 확인

### 기능 테스트
- [ ] 백테스트 실행 테스트
- [ ] 전체 종목 스캔 테스트
- [ ] 매매 상세 기록 확인
- [ ] 실시간 거래 인터페이스 확인

### 24시간 서비스 (선택사항)
- [ ] systemd 서비스 등록
- [ ] 자동 시작 설정
- [ ] 서비스 상태 확인
- [ ] 로그 확인

## 🚨 응급 상황 대처

### 서비스 완전 중지
```bash
# 모든 관련 프로세스 종료
sudo systemctl stop crypto-trading
sudo pkill -f "python.*app.py"
```

### 완전 재설치
```bash
# 프로젝트 삭제 후 재설치
rm -rf ~/crypto_trading_system
cd ~
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
# 위의 설치 과정 반복
```

### 서버 재부팅
```bash
# 서버 재부팅 (마지막 수단)
sudo reboot
```

## 📱 모바일 접속

웹 대시보드는 반응형으로 제작되어 모바일에서도 접속 가능합니다:
- 스마트폰 브라우저에서 http://34.47.77.230:5000 접속
- 모든 기능이 모바일 친화적으로 표시됩니다

## 🎯 다음 단계

웹 대시보드가 실행되면:
1. **백테스트 테스트**: 전체 종목 스캔 기능 테스트
2. **실시간 거래 설정**: 거래 전략 및 리스크 관리 설정
3. **성능 모니터링**: 시스템 리소스 사용량 모니터링
4. **자동 업데이트**: GitHub에서 자동 업데이트 설정

---

**🚀 이제 GVS 서버에서 크립토 트레이딩 시스템이 24시간 가동됩니다!**
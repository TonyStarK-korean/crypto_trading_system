# 🚀 GVS 서버 웹 대시보드 실행 가이드

## 1단계: SSH 접속

### Windows에서 SSH 접속
```bash
# 명령 프롬프트 또는 PowerShell에서 실행
ssh ubuntu@34.47.77.230
```

### SSH 키 방식 접속 (더 안전)
```bash
# SSH 키가 있는 경우
ssh -i "your-key.pem" ubuntu@34.47.77.230
```

## 2단계: 시스템 환경 확인

```bash
# 시스템 정보 확인
uname -a
python3 --version
git --version

# 네트워크 상태 확인
curl -I google.com
```

## 3단계: 프로젝트 다운로드

```bash
# 홈 디렉토리에서 시작
cd ~

# Git 저장소 클론
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git

# 프로젝트 디렉토리로 이동
cd crypto_trading_system/web_dashboard

# 파일 구조 확인
ls -la
```

## 4단계: Python 환경 설정

```bash
# Python 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 5단계: 웹 대시보드 실행

### 방법 1: 직접 실행 (테스트용)
```bash
# 가상환경이 활성화된 상태에서
python app.py
```

### 방법 2: 백그라운드 실행
```bash
# nohup을 사용하여 백그라운드 실행
nohup python app.py > app.log 2>&1 &

# 프로세스 ID 확인
ps aux | grep app.py
```

### 방법 3: Screen 사용 (권장)
```bash
# Screen 설치 (없는 경우)
sudo apt install screen

# 새로운 screen 세션 생성
screen -S crypto_trading

# 웹 대시보드 실행
python app.py

# Screen에서 빠져나오기 (Ctrl+A, D)
# 다시 연결하려면: screen -r crypto_trading
```

## 6단계: 접속 확인

### 브라우저에서 접속
- **Direct**: http://34.47.77.230:5000
- **로컬 테스트**: http://localhost:5000 (서버 내부에서)

### 연결 테스트
```bash
# 서버에서 로컬 테스트
curl http://localhost:5000

# 외부 접속 테스트
curl http://34.47.77.230:5000
```

## 7단계: 자동 시작 설정 (systemd)

### 서비스 파일 생성
```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/crypto-trading.service
```

다음 내용 입력:
```ini
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/crypto_trading_system/web_dashboard
Environment=PATH=/home/ubuntu/crypto_trading_system/web_dashboard/venv/bin
ExecStart=/home/ubuntu/crypto_trading_system/web_dashboard/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 서비스 활성화
```bash
# 서비스 등록
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading

# 서비스 시작
sudo systemctl start crypto-trading

# 서비스 상태 확인
sudo systemctl status crypto-trading
```

## 8단계: 방화벽 설정

```bash
# UFW 방화벽 설정
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw enable

# 방화벽 상태 확인
sudo ufw status
```

## 📊 문제 해결

### 포트 충돌 문제
```bash
# 포트 5000 사용 프로세스 확인
sudo lsof -i :5000

# 프로세스 종료
sudo kill -9 <PID>
```

### 권한 문제
```bash
# 프로젝트 디렉토리 권한 설정
chmod +x ~/crypto_trading_system/web_dashboard/app.py
chown -R ubuntu:ubuntu ~/crypto_trading_system
```

### 패키지 설치 실패
```bash
# 시스템 패키지 업데이트
sudo apt update
sudo apt install python3-dev build-essential

# 다시 Python 패키지 설치
pip install -r requirements.txt
```

### 로그 확인
```bash
# 애플리케이션 로그
tail -f ~/crypto_trading_system/web_dashboard/app.log

# 시스템 서비스 로그
sudo journalctl -u crypto-trading -f
```

## 🎯 빠른 실행 스크립트

다음 스크립트를 저장하고 실행하면 자동으로 설정됩니다:

```bash
# quick_start.sh 파일 생성
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Crypto Trading System..."

# 프로젝트 디렉토리로 이동
cd ~/crypto_trading_system/web_dashboard

# 가상환경 활성화
source venv/bin/activate

# 웹 대시보드 시작
echo "Starting web dashboard on port 5000..."
python app.py
EOF

# 실행 권한 부여
chmod +x quick_start.sh

# 실행
./quick_start.sh
```

## 🌐 접속 URL

설정 완료 후 다음 URL로 접속하세요:
- **메인 접속**: http://34.47.77.230:5000
- **백테스트**: http://34.47.77.230:5000/backtest
- **실시간 거래**: http://34.47.77.230:5000/live_trading

## ✅ 완료 체크리스트

- [ ] SSH 접속 성공
- [ ] 프로젝트 다운로드 완료
- [ ] Python 환경 설정 완료
- [ ] 패키지 설치 성공
- [ ] 웹 대시보드 실행 성공
- [ ] 브라우저 접속 확인
- [ ] 자동 시작 서비스 설정
- [ ] 방화벽 설정 완료

## 🚨 응급 상황 대처

### 서비스 중지
```bash
sudo systemctl stop crypto-trading
pkill -f "python app.py"
```

### 서비스 재시작
```bash
sudo systemctl restart crypto-trading
```

### 전체 재설정
```bash
# 프로젝트 디렉토리 삭제 후 재설치
rm -rf ~/crypto_trading_system
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
# 위의 과정 반복
```

---

**🎉 이제 GVS 서버에서 웹 대시보드가 24시간 가동됩니다!**
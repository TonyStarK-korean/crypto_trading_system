# 🚀 Crypto Trading System - 완전 배포 가이드

## 📋 배포 순서

### 1단계: 로컬에서 GitHub 업로드
```bash
# Windows에서 실행
C:\projects\crypto_trading_system\upload_to_github.bat
```

### 2단계: GVS 서버 접속
```bash
# SSH 접속
ssh ubuntu@34.47.77.230
```

### 3단계: 원클릭 설치 실행
```bash
# 서버에서 실행
curl -sSL https://raw.githubusercontent.com/TonyStarK-korean/crypto_trading_system/main/web_dashboard/install_server.sh | bash
```

또는 수동 설치:
```bash
# 저장소 클론
git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
cd crypto_trading_system/web_dashboard

# 실행 권한 부여
chmod +x install_server.sh

# 설치 실행
./install_server.sh
```

## 🌐 접속 URL
- **Direct**: http://34.47.77.230:5000
- **Nginx**: http://34.47.77.230

## 📊 서비스 관리

### 기본 명령어
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

### 자동 업데이트
```bash
# 자동 업데이트 상태 확인
sudo systemctl status crypto-auto-update

# 자동 업데이트 로그 확인
journalctl -u crypto-auto-update -f
```

## 🔧 문제 해결

### 서비스 실행 안 됨
```bash
# 상세 로그 확인
sudo journalctl -u crypto-trading -n 50

# 직접 실행 테스트
cd /opt/crypto_trading_system/web_dashboard
source venv/bin/activate
python app.py
```

### 포트 충돌
```bash
# 포트 사용 프로세스 확인
sudo lsof -i :5000

# 프로세스 종료
sudo kill -9 PID_NUMBER
```

### 방화벽 문제
```bash
# 방화벽 상태 확인
sudo ufw status

# 포트 허용
sudo ufw allow 5000/tcp
```

## 📈 성능 모니터링

### 시스템 리소스
```bash
# CPU, 메모리 사용량
htop

# 디스크 사용량
df -h

# 네트워크 연결
netstat -tulpn | grep :5000
```

### 애플리케이션 로그
```bash
# 애플리케이션 로그
tail -f /opt/crypto_trading_system/web_dashboard/logs/app.log

# 자동 업데이트 로그
tail -f /opt/crypto_trading_system/web_dashboard/logs/auto_update.log
```

## 🔄 업데이트 방법

### 자동 업데이트 (권장)
- 시스템이 5분마다 자동으로 GitHub에서 업데이트 확인
- 새로운 커밋이 있으면 자동으로 업데이트 및 재시작

### 수동 업데이트
```bash
cd /opt/crypto_trading_system
git pull origin main
cd web_dashboard
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart crypto-trading
```

## 🛡️ 보안 설정

### 방화벽 설정
```bash
# 기본 보안 설정
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw enable
```

### SSH 보안 강화
```bash
# SSH 설정 편집
sudo nano /etc/ssh/sshd_config

# 다음 설정 추가/수정
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no

# SSH 서비스 재시작
sudo systemctl restart sshd
```

## 📊 백테스트 전체 종목 스캔

### 사용 방법
1. 웹 대시보드 접속: http://34.47.77.230:5000
2. "백테스트" 메뉴 클릭
3. "전체 심볼 스캔" 선택
4. 원하는 전략 및 기간 설정
5. "백테스트 시작" 클릭
6. 200+ 종목 스캔 진행상황 확인

### 성능 최적화
- 메모리: 최소 4GB RAM 권장
- CPU: 멀티코어 프로세서 권장
- 네트워크: 안정적인 인터넷 연결 필수

## 🎯 24시간 운영 체크리스트

### 설치 완료 후 확인사항
- [ ] 웹 대시보드 접속 가능
- [ ] 백테스트 실행 테스트
- [ ] 실시간 거래 페이지 확인
- [ ] 서비스 자동 시작 확인
- [ ] 자동 업데이트 작동 확인
- [ ] 로그 파일 생성 확인

### 정기 점검 항목
- [ ] 시스템 리소스 사용량
- [ ] 로그 파일 크기 관리
- [ ] 백업 파일 정리
- [ ] 보안 업데이트 확인
- [ ] 네트워크 연결 상태

## 📞 지원 및 문제 해결

### 로그 파일 위치
- 애플리케이션 로그: `/opt/crypto_trading_system/web_dashboard/logs/`
- 시스템 로그: `journalctl -u crypto-trading`
- Nginx 로그: `/var/log/nginx/`

### 긴급 복구
```bash
# 전체 시스템 재시작
sudo systemctl restart crypto-trading nginx

# 백업에서 복원
sudo cp -r /opt/crypto_trading_system_backup_* /opt/crypto_trading_system
sudo systemctl restart crypto-trading
```

---

**🎉 이제 크립토 트레이딩 시스템이 GVS 서버에서 24시간 가동됩니다!**

**접속 URL**: http://34.47.77.230:5000
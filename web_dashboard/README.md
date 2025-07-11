# 🚀 Crypto Trading System Web Dashboard

실시간 암호화폐 거래 시스템 웹 대시보드

## 🎯 주요 기능

### 📊 백테스트 시스템
- **전체 종목 스캔**: 바이낸스 USDT 전체 200+ 종목 자동 스캔
- **전략별 최적화**: 이동평균, RSI, 볼린저 밴드, MACD, 스캘핑 전략
- **실시간 진행상황**: 다운로드 속도, 예상 시간, 완료율 표시
- **매매 상세 기록**: 거래별 진입/종료 시간, 레버리지, 수익률 분석
- **시장 국면 분석**: 상승장/하락장/횡보장 비율 및 변동성 분석

### 📈 실시간 거래 시스템
- **24시간 자동매매**: 무인 자동 거래 시스템
- **포지션 관리**: 실시간 포지션 모니터링 및 제어
- **위험 관리**: 손절매, 익절매, 트레일링 스탑
- **실시간 차트**: 인터랙티브 가격 차트 및 지표

### 🌐 웹 대시보드
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 지원
- **실시간 업데이트**: WebSocket 기반 실시간 데이터
- **사용자 친화적**: 직관적인 인터페이스 및 시각화

## 🛠️ 설치 및 실행

### 로컬 실행
```bash
# 저장소 클론
git clone https://github.com/your-username/crypto_trading_system.git
cd crypto_trading_system/web_dashboard

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python app.py
```

### GVS 서버 배포
```bash
# 자동 배포 스크립트 실행
chmod +x deploy.sh
./deploy.sh

# 또는 Docker 사용
docker-compose up -d
```

## 🌐 접속 URL

### 로컬 환경
- **로컬 접속**: http://127.0.0.1:5000
- **네트워크 접속**: http://[로컬IP]:5000

### GVS 서버
- **Direct**: http://34.47.77.230:5000
- **Nginx**: http://34.47.77.230

## 📋 사용법

### 1. 백테스트 실행
1. "백테스트" 메뉴 클릭
2. 단일 심볼 또는 전체 종목 스캔 선택
3. 거래 전략 선택 (자동 시간 프레임 설정)
4. 백테스트 기간 설정
5. "백테스트 시작" 클릭
6. 실시간 진행상황 확인
7. 결과 분석 및 상세 매매 기록 확인

### 2. 실시간 거래
1. "실시간 거래" 메뉴 클릭
2. 거래 설정 및 위험 관리 설정
3. 활성 전략 선택
4. "시작" 버튼으로 거래 시작
5. 실시간 모니터링 및 제어

### 3. 매매 상세 기록
1. 백테스트 기록에서 "상세 보기" 클릭
2. 시장 국면 분석 확인
3. 매매별 상세 정보 확인
4. 통계 분석 및 Excel 다운로드

## 🔧 시스템 관리

### 서비스 관리
```bash
# 서비스 상태 확인
sudo systemctl status crypto-trading

# 서비스 재시작
sudo systemctl restart crypto-trading

# 로그 확인
journalctl -u crypto-trading -f
```

### 자동 업데이트
```bash
# 자동 업데이트 실행
python auto_update.py --daemon

# 수동 업데이트
git pull origin main
sudo systemctl restart crypto-trading
```

### 네트워크 문제 해결
```bash
# 네트워크 설정 도구 실행
python network_setup.py

# 포트 확인
netstat -an | grep :5000
```

## 🚨 문제 해결

### 외부 접속 불가
1. **방화벽 설정**: 포트 5000 허용
2. **라우터 설정**: 포트 포워딩 설정
3. **서비스 상태**: 웹서버 실행 상태 확인

### 백테스트 실행 실패
1. **인터넷 연결**: 바이낸스 API 접속 확인
2. **메모리 부족**: 전체 종목 스캔 시 메모리 확인
3. **권한 문제**: 데이터베이스 쓰기 권한 확인

### 실시간 거래 문제
1. **API 키 설정**: 거래소 API 키 설정
2. **잔고 확인**: 거래 가능한 잔고 확인
3. **네트워크 상태**: 실시간 데이터 수신 상태 확인

## 📊 성능 최적화

### 전체 종목 스캔
- **병렬 처리**: 멀티스레딩으로 성능 향상
- **메모리 관리**: 대용량 데이터 처리 최적화
- **API 제한**: 바이낸스 API 호출 제한 준수

### 실시간 데이터
- **WebSocket**: 실시간 데이터 수신 최적화
- **캐싱**: 자주 사용되는 데이터 캐싱
- **압축**: 데이터 전송 압축

## 🔒 보안 설정

### 서버 보안
```bash
# 방화벽 설정
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp

# SSH 보안
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes
```

### 애플리케이션 보안
- **환경 변수**: API 키 환경 변수 저장
- **HTTPS**: SSL/TLS 인증서 설정
- **접근 제어**: IP 기반 접근 제어

## 📈 모니터링

### 시스템 모니터링
```bash
# 시스템 리소스
htop
df -h
free -h

# 네트워크 상태
netstat -an | grep :5000
```

### 애플리케이션 모니터링
- **로그 모니터링**: 실시간 로그 확인
- **성능 지표**: CPU, 메모리 사용량
- **오류 추적**: 에러 로그 분석

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License

## 📞 지원

- **이슈 리포트**: GitHub Issues
- **기능 제안**: GitHub Discussions
- **문서**: Wiki 페이지

---

**🎯 Crypto Trading System - 전문적인 암호화폐 거래 솔루션**
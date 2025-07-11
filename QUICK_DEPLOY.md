# 🚀 빠른 배포 가이드

## 1단계: GitHub 업로드 (로컬 PC)

Windows 명령 프롬프트를 **관리자 권한**으로 실행하고 다음 명령어 실행:

```bash
cd /d "C:\projects\crypto_trading_system"
git init
git add .
git commit -m "Initial commit: Complete crypto trading system with web dashboard"
git remote add origin https://github.com/TonyStarK-korean/crypto_trading_system.git
git branch -M main
git push -u origin main
```

## 2단계: 서버 배포 (GVS 서버)

SSH로 서버에 접속:
```bash
ssh ubuntu@34.47.77.230
```

원클릭 설치 실행:
```bash
curl -sSL https://raw.githubusercontent.com/TonyStarK-korean/crypto_trading_system/main/web_dashboard/install_server.sh | bash
```

## 3단계: 접속 확인

웹 브라우저에서 다음 URL로 접속:
- http://34.47.77.230:5000 (Direct)
- http://34.47.77.230 (Nginx)

## 🎯 완료!

이제 크립토 트레이딩 시스템이 24시간 가동됩니다:

✅ **바이낸스 USDT 전체 종목 스캔 백테스트**
✅ **실시간 거래 시스템**
✅ **자동 업데이트**
✅ **24시간 서비스**
✅ **웹 대시보드**

### 📊 주요 기능 테스트
1. "백테스트" → "전체 심볼 스캔" 선택
2. 원하는 전략 및 기간 설정
3. 200+ 종목 자동 스캔 실행
4. 매매 상세 기록 확인

### 🔧 서비스 관리
```bash
# 서비스 상태 확인
sudo systemctl status crypto-trading

# 로그 확인
journalctl -u crypto-trading -f

# 서비스 재시작
sudo systemctl restart crypto-trading
```

### 🎉 성공!
**여러분의 크립토 트레이딩 시스템이 GVS 서버에서 24시간 가동 중입니다!**
# 🚀 크립토 트레이딩 시스템 빠른 시작 가이드

## 현재 상태
웹 대시보드가 성공적으로 실행 중입니다!

## 📍 접속 URL
- **로컬 접속**: http://127.0.0.1:5000
- **네트워크 접속**: http://10.81.164.73:5000

## 🎯 주요 기능 테스트

### 1. 메인 대시보드
- 백테스트 vs 실시간 거래 선택
- 실시간 가격 모니터링
- 시스템 상태 확인

### 2. 백테스트 시스템
- 거래 전략 선택
- 초기 자본금 설정
- 백테스트 실행 및 결과 확인

### 3. 실시간 거래 시스템
- 거래 시작/중지 제어
- 실시간 포지션 관리
- 거래 설정 조정

## 🔧 GVS 서버 배포 (34.47.77.230:5000)

### 1단계: SSH 접속
```bash
ssh ubuntu@34.47.77.230
```

### 2단계: 서버 설정
```bash
# 서버 설정 스크립트 실행
sudo apt update && sudo apt install -y python3 python3-pip
pip3 install Flask pandas numpy ccxt plotly gunicorn
```

### 3단계: 애플리케이션 배포
```bash
# 애플리케이션 실행
python3 app.py
```

### 4단계: 영구 실행 설정
```bash
# 백그라운드 실행
nohup python3 app.py > server.log 2>&1 &
```

## 📊 테스트 체크리스트
- [ ] 메인 페이지 로드 확인
- [ ] 백테스트 실행 테스트
- [ ] 실시간 거래 시뮬레이션 테스트
- [ ] 차트 표시 확인
- [ ] 반응형 디자인 확인

## 🎨 디자인 특징
- 모던하고 세련된 UI
- 반응형 디자인 (모바일 지원)
- 실시간 데이터 업데이트
- 인터랙티브 차트

## 🔒 보안 설정
- 기본적인 보안 설정 적용
- 환경변수 기반 설정
- 데이터베이스 보안 연결

---
**시작 시간**: 2025-07-11
**상태**: 실행 중 ✅
**포트**: 5000
**환경**: Development Mode
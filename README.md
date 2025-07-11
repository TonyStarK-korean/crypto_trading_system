# Crypto Trading System

전세계 상위 0.01%급 코인선물 자동매매 시스템

## 🚀 주요 기능

- **백테스트 엔진**: 전략 수익률 검증
- **ML 예측 시스템**: 확률 기반 매매 신호 생성
- **복리 시스템**: 자동 복리 계산
- **리스크 관리**: 동적 레버리지, 비중, 익절, 손절
- **실시간 거래**: 24시간 자동매매
- **웹 대시보드**: 실시간 모니터링 (추후 확장)

## 📁 프로젝트 구조

```
crypto_trading_system/
├── src/
│   ├── core/               # 핵심 시스템
│   ├── strategies/         # 전략 모듈들
│   ├── ml/                # ML 예측 시스템
│   ├── backtest/          # 백테스트 엔진 (Backtrader 포함)
│   ├── database/          # 데이터베이스 관리 (PostgreSQL)
│   ├── realtime/          # 실시간 데이터 처리 (WebSocket)
│   ├── data_collection/   # 자동화된 데이터 수집
│   ├── monitoring/        # 성능 모니터링 시스템
│   └── utils/             # 유틸리티
├── data/                  # 데이터 저장소
├── config/                # 설정 파일들
├── results/               # 백테스트 결과
├── logs/                  # 로그 파일들
├── integrated_system.py   # 통합 시스템 실행 파일
└── docs/                  # 문서
```

## 🛠 설치 및 실행

1. **환경 설정**
```bash
pip install -r requirements.txt
```

2. **설정 파일 생성**
```bash
cp config/config.example.yaml config/config.yaml
# config.yaml 파일에서 API 키 및 설정 수정
```

3. **백테스트 실행**
```bash
python src/backtest/main.py
```

4. **통합 시스템 실행**
```bash
python integrated_system.py
```

5. **개별 구성 요소 실행**
```bash
# 백테스트 엔진 (Backtrader)
python src/backtest/backtrader_engine.py

# 실시간 데이터 수집
python src/realtime/websocket_client.py

# 자동화된 데이터 수집
python src/data_collection/data_collector.py

# 성능 모니터링
python src/monitoring/performance_monitor.py
```

## 📊 백테스트 결과

- 수익률 분석
- 리스크 지표
- 최대 낙폭 (MDD)
- 샤프 비율
- 승률 및 손익비

## 🔧 새로 추가된 기능

- [x] **PostgreSQL 데이터베이스 시스템** - 실시간 데이터 저장 및 관리
- [x] **WebSocket 실시간 데이터 처리** - 바이낸스 실시간 데이터 스트리밍
- [x] **Backtrader 백테스트 프레임워크** - 고도화된 백테스트 엔진
- [x] **자동화된 데이터 수집** - 다중 거래소 데이터 자동 수집
- [x] **성능 모니터링 시스템** - 시스템 및 거래 성능 실시간 모니터링
- [x] **통합 거래 시스템** - 모든 구성 요소를 하나로 연결

## 🔧 확장 예정 기능

- [ ] 동적 레버리지 시스템
- [ ] 다중 전략 지원
- [ ] 웹 대시보드
- [ ] 실시간 알림
- [ ] 포트폴리오 최적화

## 📝 라이선스

MIT License 
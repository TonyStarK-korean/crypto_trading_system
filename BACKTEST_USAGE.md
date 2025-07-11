# 백테스트 실행 가이드

## 개요
이 프로젝트는 암호화폐 거래 전략을 백테스트할 수 있는 시스템입니다. 다양한 전략과 설정으로 백테스트를 실행할 수 있습니다.

## 백테스트 실행 파일들

### 1. run_backtest.py (고급 백테스트)
가장 완전한 기능을 제공하는 백테스트 실행 파일입니다.

#### 사용법
```bash
# 기본 실행
python run_backtest.py

# 옵션 지정
python run_backtest.py --symbol BTC/USDT --strategy simple --start-date 2023-01-01 --end-date 2023-12-31

# 다른 전략 사용
python run_backtest.py --strategy bollinger_breakout --initial-balance 50000

# 다른 심볼과 시간프레임
python run_backtest.py --symbol ETH/USDT --timeframe 4h --start-date 2023-06-01 --end-date 2023-12-31
```

#### 옵션 설명
- `--symbol, -s`: 거래 심볼 (기본값: BTC/USDT)
- `--strategy, -st`: 백테스트 전략 (simple, bollinger_breakout)
- `--start-date, -sd`: 시작 날짜 (YYYY-MM-DD)
- `--end-date, -ed`: 종료 날짜 (YYYY-MM-DD)
- `--timeframe, -t`: 시간프레임 (1h, 4h, 1d 등)
- `--initial-balance, -ib`: 초기 자본
- `--commission, -c`: 수수료
- `--config, -cf`: 설정 파일 경로

### 2. quick_backtest.py (간단한 백테스트)
기본 설정으로 빠르게 백테스트를 실행할 수 있는 스크립트입니다.

#### 사용법
```bash
python quick_backtest.py
```

이 스크립트는 다음 기본 설정으로 실행됩니다:
- 심볼: BTC/USDT
- 전략: simple (이동평균 교차)
- 기간: 최근 30일
- 초기 자본: 10,000 USDT
- 수수료: 0.1%

### 3. example_backtest.py (예제 백테스트)
학습용 예제 백테스트입니다.

#### 사용법
```bash
python example_backtest.py
```

## 지원하는 전략

### 1. Simple Strategy (이동평균 교차)
- 단기 이동평균과 장기 이동평균의 교차를 이용한 전략
- RSI 필터링 포함
- 매수: 골든 크로스 + RSI < 70
- 매도: 데드 크로스 + RSI > 30

### 2. Bollinger Breakout Strategy
- 볼린저 밴드 돌파 전략
- 상단 밴드 돌파 시 매수
- 하단 밴드 돌파 시 매도

## 결과 파일

백테스트 실행 후 다음 파일들이 생성됩니다:

### 1. JSON 결과 파일
- 위치: `results/backtest_{전략명}_{타임스탬프}.json`
- 포함 내용:
  - 성과 지표 (수익률, 최대 낙폭, 샤프 비율 등)
  - 거래 내역
  - 자산 곡선

### 2. 차트 파일
- 위치: `results/backtest_{전략명}_{타임스탬프}.png`
- 포함 내용:
  - 가격 차트
  - 거래 신호
  - 자산 곡선

## 성과 지표

백테스트 결과에서 확인할 수 있는 성과 지표들:

- **총 수익률**: 전체 백테스트 기간의 수익률
- **연간 수익률**: 연간화된 수익률
- **최대 낙폭**: 최대 손실 폭
- **샤프 비율**: 위험 대비 수익률
- **승률**: 수익 거래 비율
- **총 거래 수**: 전체 거래 횟수
- **손익비**: 평균 수익 / 평균 손실

## 설정 파일

`config/config.yaml` 파일에서 백테스트 설정을 변경할 수 있습니다:

```yaml
backtest:
  initial_balance: 10000.0
  commission: 0.001
  slippage: 0.0005

strategy:
  params:
    short_period: 20
    long_period: 50
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
```

## 데이터

### 데이터 소스
- 실제 거래소 API (Binance, Bybit 등)
- 저장된 CSV 파일
- 샘플 데이터 (API 연결 실패 시)

### 데이터 저장
- 위치: `data/` 폴더
- 형식: `{심볼}_{시간프레임}_{날짜}.csv`

## 문제 해결

### 1. 데이터 수집 실패
- 인터넷 연결 확인
- API 키 설정 확인
- 샘플 데이터로 자동 대체

### 2. 모듈 import 오류
```bash
pip install -r requirements.txt
```

### 3. 결과 파일이 생성되지 않음
- `results/` 폴더 권한 확인
- 디스크 공간 확인

## 예제 실행

```bash
# 1. 간단한 백테스트
python quick_backtest.py

# 2. 고급 백테스트 (1년 데이터)
python run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31

# 3. 볼린저 밴드 전략
python run_backtest.py --strategy bollinger_breakout --initial-balance 20000

# 4. 다른 심볼로 테스트
python run_backtest.py --symbol ETH/USDT --timeframe 4h
```

## 주의사항

1. **과적합 방지**: 백테스트 결과가 좋다고 해서 실제 거래에서도 같은 성과를 보장하지는 않습니다.
2. **거래 비용**: 수수료, 슬리피지 등 실제 거래 비용을 고려해야 합니다.
3. **데이터 품질**: 사용하는 데이터의 품질과 정확성을 확인하세요.
4. **리스크 관리**: 실제 거래에서는 적절한 리스크 관리가 필수입니다. 
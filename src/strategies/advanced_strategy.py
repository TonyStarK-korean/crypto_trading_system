"""
고급 통합 전략 (Advanced Integrated Strategy)
A. 적응형 전략 (Adaptive Strategy)
B. 다중 시간프레임 전략 (Multi-Timeframe Strategy)  
C. 머신러닝 기반 전략 (ML-Based Strategy)
E. 리스크 패리티 전략 (Risk Parity Strategy)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)

class AdvancedIntegratedStrategy:
    """고급 통합 전략"""
    
    def __init__(self, 
                 # 적응형 전략 파라미터
                 adaptive_window: int = 50,
                 volatility_lookback: int = 20,
                 
                 # 다중 시간프레임 파라미터
                 timeframes: List[str] = ['1h', '4h', '1d'],
                 
                 # 머신러닝 파라미터
                 ml_enabled: bool = True,
                 ml_lookback: int = 100,
                 ml_features: int = 20,
                 
                 # 리스크 패리티 파라미터
                 risk_target: float = 0.02,
                 max_position_size: float = 0.1,
                 volatility_target: float = 0.15,
                 
                 # 고도화된 시장국면 대응 파라미터
                 enable_dynamic_leverage: bool = True,
                 enable_dynamic_position_sizing: bool = True,
                 enable_dynamic_thresholds: bool = True):
        
        # 적응형 전략 설정
        self.adaptive_window = adaptive_window
        self.volatility_lookback = volatility_lookback
        
        # 다중 시간프레임 설정
        self.timeframes = timeframes
        self.timeframe_weights = {'1h': 0.5, '4h': 0.3, '1d': 0.2}
        
        # 머신러닝 설정
        self.ml_enabled = ml_enabled
        self.ml_lookback = ml_lookback
        self.ml_features = ml_features
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 리스크 패리티 설정
        self.risk_target = risk_target
        self.max_position_size = max_position_size
        self.volatility_target = volatility_target
        
        # 고도화된 시장국면 대응 설정
        self.enable_dynamic_leverage = enable_dynamic_leverage
        self.enable_dynamic_position_sizing = enable_dynamic_position_sizing
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        
        # 상태 변수
        self.market_regime = 'neutral'  # trending, ranging, volatile
        self.current_volatility = 0.0
        self.adaptive_thresholds = {}
        self.current_leverage = 1.0
        self.dynamic_risk_multiplier = 1.0
        
        # 시장 상태별 설정 매핑
        self.regime_settings = self.initialize_regime_settings()
    
    def initialize_regime_settings(self) -> Dict:
        """시장 상태별 설정 초기화"""
        return {
            'trending': {
                'leverage_multiplier': 1.5,      # 추세장: 1.5배 레버리지
                'risk_multiplier': 1.3,          # 추세장: 리스크 30% 증가
                'entry_threshold': 0.6,          # 추세장: 낮은 진입 임계값 (적극 진입)
                'confidence_boost': 0.1,         # 추세장: 신뢰도 10% 부스트
                'max_hold_hours': 24,            # 추세장: 24시간 보유
                'trailing_multiplier': 1.2       # 추세장: 트레일링 스탑 완화
            },
            'ranging': {
                'leverage_multiplier': 0.8,      # 횡보장: 0.8배 레버리지
                'risk_multiplier': 0.7,          # 횡보장: 리스크 30% 감소
                'entry_threshold': 0.8,          # 횡보장: 높은 진입 임계값 (신중 진입)
                'confidence_boost': -0.1,        # 횡보장: 신뢰도 10% 감소
                'max_hold_hours': 8,             # 횡보장: 8시간 보유
                'trailing_multiplier': 0.8       # 횡보장: 트레일링 스탑 강화
            },
            'volatile': {
                'leverage_multiplier': 0.6,      # 변동성장: 0.6배 레버리지
                'risk_multiplier': 0.5,          # 변동성장: 리스크 50% 감소
                'entry_threshold': 0.75,         # 변동성장: 중간 진입 임계값
                'confidence_boost': -0.05,       # 변동성장: 신뢰도 5% 감소
                'max_hold_hours': 4,             # 변동성장: 4시간 보유
                'trailing_multiplier': 0.6       # 변동성장: 트레일링 스탑 매우 강화
            },
            'trending_volatile': {
                'leverage_multiplier': 1.2,      # 추세+변동성: 1.2배 레버리지
                'risk_multiplier': 1.0,          # 추세+변동성: 기본 리스크
                'entry_threshold': 0.65,         # 추세+변동성: 중저 진입 임계값
                'confidence_boost': 0.05,        # 추세+변동성: 신뢰도 5% 부스트
                'max_hold_hours': 12,            # 추세+변동성: 12시간 보유
                'trailing_multiplier': 1.0       # 추세+변동성: 기본 트레일링 스탑
            },
            'ranging_volatile': {
                'leverage_multiplier': 0.7,      # 횡보+변동성: 0.7배 레버리지
                'risk_multiplier': 0.6,          # 횡보+변동성: 리스크 40% 감소
                'entry_threshold': 0.85,         # 횡보+변동성: 매우 높은 진입 임계값
                'confidence_boost': -0.15,       # 횡보+변동성: 신뢰도 15% 감소
                'max_hold_hours': 6,             # 횡보+변동성: 6시간 보유
                'trailing_multiplier': 0.7       # 횡보+변동성: 트레일링 스탑 강화
            },
            'neutral': {
                'leverage_multiplier': 1.0,      # 중립: 기본 레버리지
                'risk_multiplier': 1.0,          # 중립: 기본 리스크
                'entry_threshold': 0.7,          # 중립: 기본 진입 임계값
                'confidence_boost': 0.0,         # 중립: 신뢰도 변화 없음
                'max_hold_hours': 12,            # 중립: 12시간 보유
                'trailing_multiplier': 1.0       # 중립: 기본 트레일링 스탑
            }
        }
        
    def calculate_market_volatility(self, df: pd.DataFrame) -> float:
        """시장 변동성 계산"""
        returns = df['close'].pct_change().dropna()
        return returns.rolling(window=self.volatility_lookback).std().iloc[-1] * np.sqrt(24)
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """시장 상태 감지"""
        # 변동성 계산
        volatility = self.calculate_market_volatility(df)
        
        # 추세 강도 계산
        price_change_20 = df['close'].pct_change(20).iloc[-1]
        price_change_5 = df['close'].pct_change(5).iloc[-1]
        
        # 볼린저 밴드 위치
        bb_position = self.calculate_bb_position(df)
        
        # 시장 상태 판단
        if volatility > 0.05:  # 고변동성
            if abs(price_change_20) > 0.2:  # 강한 추세
                return 'trending_volatile'
            else:
                return 'ranging_volatile'
        elif abs(price_change_20) > 0.15:  # 중간 변동성, 강한 추세
            return 'trending'
        elif abs(price_change_5) < 0.02:  # 낮은 변동성, 약한 추세
            return 'ranging'
        else:
            return 'neutral'
    
    def calculate_bb_position(self, df: pd.DataFrame) -> float:
        """볼린저 밴드 내 위치 계산 (0-1)"""
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if bb_upper == bb_lower:
            return 0.5
        
        position = (current_price - bb_lower) / (bb_upper - bb_lower)
        return np.clip(position, 0, 1)
    
    def update_adaptive_thresholds(self, df: pd.DataFrame):
        """적응형 임계값 업데이트 (고도화된 시장국면 대응)"""
        volatility = self.calculate_market_volatility(df)
        self.current_volatility = volatility
        
        # 변동성 기반 임계값 조정
        base_volatility = 0.03  # 기준 변동성
        volatility_multiplier = volatility / base_volatility
        
        # 시장 상태별 설정 가져오기
        regime_settings = self.regime_settings.get(self.market_regime, self.regime_settings['neutral'])
        
        # 기존 regime_multipliers를 regime_settings로 대체
        regime_mult = regime_settings.get('risk_multiplier', 1.0)
        
        # 레버리지 업데이트
        if self.enable_dynamic_leverage:
            self.current_leverage = regime_settings.get('leverage_multiplier', 1.0)
        
        # 리스크 배수 업데이트
        if self.enable_dynamic_position_sizing:
            self.dynamic_risk_multiplier = regime_mult
        
        # 적응형 임계값 설정 (기존 + 고도화)
        self.adaptive_thresholds = {
            'volume_threshold': 1.5 + (volatility_multiplier * 0.5),
            'price_threshold': 0.02 * volatility_multiplier * regime_mult,
            'momentum_threshold': 0.01 * volatility_multiplier,
            'rsi_overbought': 70 + (volatility_multiplier * 10),
            'rsi_oversold': 30 - (volatility_multiplier * 10),
            'stop_loss': 0.015 * volatility_multiplier * regime_mult,
            'take_profit': 0.04 * volatility_multiplier * regime_mult,
            
            # 고도화된 임계값 추가
            'entry_threshold': regime_settings.get('entry_threshold', 0.7),
            'confidence_boost': regime_settings.get('confidence_boost', 0.0),
            'max_hold_hours': regime_settings.get('max_hold_hours', 12),
            'trailing_multiplier': regime_settings.get('trailing_multiplier', 1.0)
        }
    
    def calculate_multi_timeframe_indicators(self, df_1h: pd.DataFrame, 
                                           df_4h: pd.DataFrame = None, 
                                           df_1d: pd.DataFrame = None) -> Dict:
        """다중 시간프레임 지표 계산"""
        indicators = {}
        
        # 1시간봉 지표
        indicators['1h'] = self.calculate_single_timeframe_indicators(df_1h, '1h')
        
        # 4시간봉 지표 (1시간봉에서 리샘플링)
        if df_4h is None:
            df_4h = self.resample_data(df_1h, '4h')
        indicators['4h'] = self.calculate_single_timeframe_indicators(df_4h, '4h')
        
        # 1일봉 지표 (1시간봉에서 리샘플링)
        if df_1d is None:
            df_1d = self.resample_data(df_1h, '1d')
        indicators['1d'] = self.calculate_single_timeframe_indicators(df_1d, '1d')
        
        return indicators
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        rule_map = {'4h': '4H', '1d': '1D'}
        rule = rule_map.get(timeframe, '1H')
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_single_timeframe_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """단일 시간프레임 지표 계산"""
        # 기본 지표
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # 이동평균
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['ma_20'] + (bb_std * 2)
        df['bb_lower'] = df['ma_20'] - (bb_std * 2)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 거래량 지표
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 모멘텀 지표
        df['momentum'] = df['close'].pct_change(10)
        df['roc'] = df['close'].pct_change(20)
        
        # 현재 상태 반환
        latest = df.iloc[-1]
        
        return {
            'price': latest['close'],
            'ma_10': latest['ma_10'],
            'ma_20': latest['ma_20'],
            'ma_50': latest['ma_50'],
            'rsi': latest['rsi'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'macd_histogram': latest['macd_histogram'],
            'volume_ratio': latest['volume_ratio'],
            'momentum': latest['momentum'],
            'roc': latest['roc'],
            'volatility': latest['volatility'],
            'bb_position': self.calculate_bb_position(df)
        }
    
    def prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """머신러닝 특성 준비"""
        features = []
        
        # 가격 기반 특성
        features.append(df['close'].pct_change(1).fillna(0))  # 1시간 수익률
        features.append(df['close'].pct_change(4).fillna(0))  # 4시간 수익률
        features.append(df['close'].pct_change(24).fillna(0))  # 1일 수익률
        
        # 기술적 지표
        features.append(df['rsi'].fillna(50))
        features.append(df['macd'].fillna(0))
        features.append(df['macd_histogram'].fillna(0))
        features.append(df['volume_ratio'].fillna(1))
        features.append(df['momentum'].fillna(0))
        
        # 이동평균 관계
        features.append((df['close'] / df['ma_10']).fillna(1))
        features.append((df['close'] / df['ma_20']).fillna(1))
        features.append((df['close'] / df['ma_50']).fillna(1))
        features.append((df['ma_10'] / df['ma_20']).fillna(1))
        
        # 볼린저 밴드 관계
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['ma_20']
        features.append(bb_width.fillna(0))
        
        # 변동성 지표
        features.append(df['volatility'].fillna(0))
        
        # 거래량 변화
        features.append(df['volume'].pct_change().fillna(0))
        
        # 추가 기술적 지표들
        features.append(df['roc'].fillna(0))
        
        # 시간 기반 특성
        features.append(df.index.hour / 24)  # 시간 정규화
        features.append(df.index.dayofweek / 7)  # 요일 정규화
        
        # 특성 행렬 생성
        feature_matrix = np.column_stack(features)
        
        return feature_matrix
    
    def train_ml_model(self, df: pd.DataFrame):
        """머신러닝 모델 훈련"""
        if not self.ml_enabled or len(df) < self.ml_lookback:
            return
        
        # 특성 준비
        features = self.prepare_ml_features(df)
        
        # 타겟 생성 (다음 시간 가격 상승 여부)
        future_returns = df['close'].pct_change().shift(-1)
        targets = (future_returns > 0.01).astype(int)  # 1% 이상 상승
        
        # 유효한 데이터만 사용
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        if len(features) < 50:  # 최소 데이터 요구사항
            return
        
        # 훈련/검증 분할
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 훈련
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # 성능 평가
        train_score = self.ml_model.score(X_train_scaled, y_train)
        test_score = self.ml_model.score(X_test_scaled, y_test)
        
        logger.info(f"ML 모델 훈련 완료 - 훈련 정확도: {train_score:.3f}, 검증 정확도: {test_score:.3f}")
        
        self.is_trained = True
    
    def get_ml_prediction(self, df: pd.DataFrame) -> float:
        """머신러닝 예측"""
        if not self.ml_enabled or not self.is_trained or self.ml_model is None:
            return 0.5  # 중립
        
        try:
            features = self.prepare_ml_features(df)
            latest_features = features[-1:].reshape(1, -1)
            
            if np.isnan(latest_features).any():
                return 0.5
            
            scaled_features = self.scaler.transform(latest_features)
            prediction_proba = self.ml_model.predict_proba(scaled_features)[0]
            
            return prediction_proba[1]  # 상승 확률 반환
            
        except Exception as e:
            logger.error(f"ML 예측 오류: {e}")
            return 0.5
    
    def calculate_risk_parity_position_size(self, df: pd.DataFrame, balance: float) -> Tuple[float, float]:
        """리스크 패리티 기반 포지션 크기 계산 (고도화된 버전)"""
        # 현재 변동성 계산
        volatility = self.calculate_market_volatility(df)
        
        # 타겟 변동성 대비 현재 변동성
        volatility_ratio = self.volatility_target / max(volatility, 0.01)
        
        # 기본 포지션 크기
        base_position = balance * self.risk_target
        
        # 고도화된 시장국면 대응
        if self.enable_dynamic_position_sizing:
            # 시장 상태별 리스크 조정
            base_position *= self.dynamic_risk_multiplier
            
        # 변동성 조정 포지션 크기
        adjusted_position = base_position * volatility_ratio
        
        # 레버리지 적용
        leverage = self.current_leverage if self.enable_dynamic_leverage else 1.0
        leveraged_position = adjusted_position * leverage
        
        # 최대 포지션 크기 제한 (레버리지 고려)
        max_position = balance * self.max_position_size * leverage
        
        final_position = min(leveraged_position, max_position)
        
        return final_position, leverage
    
    def calculate_dynamic_exit_levels(self, entry_price: float, df: pd.DataFrame) -> Dict:
        """동적 청산 레벨 계산 (고도화된 버전)"""
        volatility = self.current_volatility
        
        # 기본 레벨
        base_stop = self.adaptive_thresholds.get('stop_loss', 0.02)
        base_take = self.adaptive_thresholds.get('take_profit', 0.05)
        
        # ATR 기반 조정
        atr = self.calculate_atr(df, 14)
        atr_multiplier = atr / entry_price
        
        # 시장 상태별 트레일링 스탑 조정
        trailing_multiplier = self.adaptive_thresholds.get('trailing_multiplier', 1.0)
        
        # 동적 레벨 계산 (시장 상태 반영)
        dynamic_stop = entry_price * (1 - max(base_stop, atr_multiplier * 2))
        dynamic_take = entry_price * (1 + max(base_take, atr_multiplier * 3))
        trailing_stop = entry_price * (1 - atr_multiplier * 1.5 * trailing_multiplier)
        
        return {
            'stop_loss': dynamic_stop,
            'take_profit': dynamic_take,
            'trailing_stop': trailing_stop,
            'leverage': self.current_leverage,
            'market_regime': self.market_regime,
            'atr_multiplier': atr_multiplier,
            'trailing_multiplier': trailing_multiplier
        }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR (Average True Range) 계산"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.02
    
    def calculate_multi_timeframe_signal(self, indicators: Dict) -> float:
        """다중 시간프레임 신호 계산"""
        signals = {}
        
        for timeframe, data in indicators.items():
            signal = 0.0
            
            # 추세 신호
            if data['price'] > data['ma_20']:
                signal += 1
            if data['ma_10'] > data['ma_20']:
                signal += 1
            if data['ma_20'] > data['ma_50']:
                signal += 1
            
            # 모멘텀 신호
            if data['rsi'] > 30 and data['rsi'] < 70:
                signal += 1
            if data['macd'] > data['macd_signal']:
                signal += 1
            if data['momentum'] > 0:
                signal += 1
            
            # 볼린저 밴드 신호
            if data['bb_position'] > 0.2 and data['bb_position'] < 0.8:
                signal += 1
            
            # 거래량 신호
            if data['volume_ratio'] > 1.2:
                signal += 1
            
            signals[timeframe] = signal / 8  # 정규화
        
        # 가중평균 계산
        weighted_signal = sum(
            signals[tf] * self.timeframe_weights[tf] 
            for tf in signals.keys()
        )
        
        return weighted_signal
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int, 
                       position: Optional[Dict] = None, balance: float = 0) -> Tuple[Optional[str], Optional[Dict]]:
        """통합 신호 생성"""
        try:
            # 충분한 데이터 확인
            if current_idx < 100:
                return None, position
            
            # 현재 데이터 슬라이스
            current_df = df.iloc[:current_idx+1].copy()
            
            # 기술적 지표 계산
            current_df = self.calculate_technical_indicators(current_df)
            
            # 시장 상태 감지
            self.market_regime = self.detect_market_regime(current_df)
            
            # 적응형 임계값 업데이트
            self.update_adaptive_thresholds(current_df)
            
            # 머신러닝 모델 훈련 (주기적)
            if current_idx % 100 == 0:  # 100봉마다 재훈련
                self.train_ml_model(current_df)
            
            # 다중 시간프레임 지표 계산
            indicators = self.calculate_multi_timeframe_indicators(current_df)
            
            # 포지션이 없을 때 진입 신호 확인
            if position is None:
                # 다중 시간프레임 신호
                mtf_signal = self.calculate_multi_timeframe_signal(indicators)
                
                # 머신러닝 신호
                ml_signal = self.get_ml_prediction(current_df)
                
                # 통합 신호 계산
                combined_signal = (mtf_signal * 0.6) + (ml_signal * 0.4)
                
                # 고도화된 진입 조건 확인
                entry_threshold = self.adaptive_thresholds.get('entry_threshold', 0.7)
                confidence_boost = self.adaptive_thresholds.get('confidence_boost', 0.0)
                
                # 신뢰도 부스트 적용
                adjusted_signal = combined_signal + confidence_boost
                adjusted_signal = max(0, min(1, adjusted_signal))  # 0-1 범위로 제한
                
                # 동적 진입 조건 확인
                if self.enable_dynamic_thresholds:
                    signal_condition = adjusted_signal > entry_threshold
                else:
                    signal_condition = combined_signal > 0.7  # 기본 임계값
                
                if signal_condition:  # 동적 매수 신호
                    entry_price = current_df['close'].iloc[-1]
                    position_size, leverage = self.calculate_risk_parity_position_size(current_df, balance)
                    
                    # 동적 청산 레벨 계산
                    exit_levels = self.calculate_dynamic_exit_levels(entry_price, current_df)
                    
                    new_position = {
                        'entry_price': entry_price,
                        'entry_time': current_df.index[-1],
                        'size': position_size,
                        'stop_loss': exit_levels['stop_loss'],
                        'take_profit': exit_levels['take_profit'],
                        'trailing_stop': exit_levels['trailing_stop'],
                        'highest_price': entry_price,
                        'strategy_type': 'integrated',
                        'market_regime': self.market_regime,
                        'ml_confidence': ml_signal,
                        'mtf_signal': mtf_signal,
                        
                        # 고도화된 정보 추가
                        'leverage': leverage,
                        'risk_multiplier': self.dynamic_risk_multiplier,
                        'entry_threshold': entry_threshold,
                        'confidence_boost': confidence_boost,
                        'adjusted_signal': adjusted_signal,
                        'combined_signal': combined_signal,
                        'volatility': self.current_volatility
                    }
                    
                    return 'buy', new_position
            
            # 포지션이 있을 때 청산 조건 확인
            elif position is not None:
                current_price = current_df['close'].iloc[-1]
                signal = self.check_exit_conditions(position, current_price, current_df)
                
                if signal == 'sell':
                    return 'sell', position
            
            return None, position
            
        except Exception as e:
            logger.error(f"신호 생성 오류: {e}")
            return None, position
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        # 이동평균
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['ma_20'] + (bb_std * 2)
        df['bb_lower'] = df['ma_20'] - (bb_std * 2)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 거래량 지표
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 모멘텀 지표
        df['momentum'] = df['close'].pct_change(10)
        df['roc'] = df['close'].pct_change(20)
        
        return df
    
    def check_exit_conditions(self, position: Dict, current_price: float, df: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        entry_price = position['entry_price']
        current_profit_rate = (current_price - entry_price) / entry_price
        
        # 고점 업데이트
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
            # 트레일링 스탑 업데이트
            atr = self.calculate_atr(df, 14)
            position['trailing_stop'] = current_price * (1 - (atr / current_price) * 1.5)
        
        # 손절 조건
        if current_price <= position['stop_loss']:
            return 'sell'
        
        # 익절 조건
        if current_price >= position['take_profit']:
            return 'sell'
        
        # 트레일링 스탑
        if current_price <= position['trailing_stop'] and current_profit_rate > 0:
            return 'sell'
        
        # 시장 상태 변화 기반 청산
        if self.market_regime != position.get('market_regime', 'neutral'):
            # 시장 상태가 변했을 때 머신러닝 신호 확인
            ml_signal = self.get_ml_prediction(df)
            if ml_signal < 0.3:  # 약한 매도 신호
                return 'sell'
        
        # 시간 기반 청산 (고도화된 시장 상태별 조정)
        hold_time = df.index[-1] - position['entry_time']
        hold_hours = hold_time.total_seconds() / 3600
        
        # 동적 최대 보유시간 적용
        max_hold_time = self.adaptive_thresholds.get('max_hold_hours', 12)
        
        if hold_hours >= max_hold_time:
            return 'sell'
        
        return None
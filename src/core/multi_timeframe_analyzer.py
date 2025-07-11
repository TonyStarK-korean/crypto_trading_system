"""
다중 시간프레임 분석기
1시간, 4시간, 1일봉 데이터를 통합 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """다중 시간프레임 분석기"""
    
    def __init__(self, 
                 timeframes: List[str] = ['1h', '4h', '1d'],
                 weights: Dict[str, float] = None):
        
        self.timeframes = timeframes
        self.weights = weights or {'1h': 0.5, '4h': 0.3, '1d': 0.2}
        self.indicators_cache = {}
        
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        if df.empty:
            return df
            
        # 시간프레임 매핑
        resample_map = {
            '1h': '1H',
            '4h': '4H', 
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        
        rule = resample_map.get(target_timeframe, '1H')
        
        try:
            # 리샘플링 수행
            resampled = df.resample(rule, label='right', closed='right').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"데이터 리샘플링 오류 ({target_timeframe}): {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """기술적 지표 계산"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 이동평균선
        periods = self.get_ma_periods(timeframe)
        for period in periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        rsi_period = self.get_rsi_period(timeframe)
        df['rsi'] = self.calculate_rsi(df['close'], rsi_period)
        
        # MACD
        macd_fast, macd_slow, macd_signal = self.get_macd_periods(timeframe)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(
            df['close'], macd_fast, macd_slow, macd_signal
        )
        
        # 볼린저 밴드
        bb_period, bb_std = self.get_bollinger_periods(timeframe)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], bb_period, bb_std
        )
        
        # Stochastic
        stoch_k, stoch_d = self.get_stochastic_periods(timeframe)
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(
            df['high'], df['low'], df['close'], stoch_k, stoch_d
        )
        
        # Williams %R
        williams_period = self.get_williams_period(timeframe)
        df['williams_r'] = self.calculate_williams_r(
            df['high'], df['low'], df['close'], williams_period
        )
        
        # ATR (Average True Range)
        atr_period = self.get_atr_period(timeframe)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], atr_period)
        
        # 거래량 지표
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 모멘텀 지표
        momentum_period = self.get_momentum_period(timeframe)
        df['momentum'] = df['close'].pct_change(momentum_period)
        df['roc'] = df['close'].pct_change(momentum_period) * 100
        
        # 추세 지표
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])
        
        # 지지/저항 레벨
        df['pivot'], df['s1'], df['s2'], df['r1'], df['r2'] = self.calculate_pivot_points(
            df['high'], df['low'], df['close']
        )
        
        return df
    
    def get_ma_periods(self, timeframe: str) -> List[int]:
        """시간프레임별 이동평균 기간"""
        periods_map = {
            '1h': [10, 20, 50, 100, 200],
            '4h': [10, 20, 50, 100],
            '1d': [10, 20, 50, 100, 200],
            '1w': [10, 20, 50],
            '1M': [10, 20]
        }
        return periods_map.get(timeframe, [10, 20, 50])
    
    def get_rsi_period(self, timeframe: str) -> int:
        """시간프레임별 RSI 기간"""
        periods_map = {
            '1h': 14,
            '4h': 14,
            '1d': 14,
            '1w': 14,
            '1M': 14
        }
        return periods_map.get(timeframe, 14)
    
    def get_macd_periods(self, timeframe: str) -> Tuple[int, int, int]:
        """시간프레임별 MACD 기간"""
        periods_map = {
            '1h': (12, 26, 9),
            '4h': (12, 26, 9),
            '1d': (12, 26, 9),
            '1w': (12, 26, 9),
            '1M': (12, 26, 9)
        }
        return periods_map.get(timeframe, (12, 26, 9))
    
    def get_bollinger_periods(self, timeframe: str) -> Tuple[int, float]:
        """시간프레임별 볼린저 밴드 기간"""
        periods_map = {
            '1h': (20, 2.0),
            '4h': (20, 2.0),
            '1d': (20, 2.0),
            '1w': (20, 2.0),
            '1M': (20, 2.0)
        }
        return periods_map.get(timeframe, (20, 2.0))
    
    def get_stochastic_periods(self, timeframe: str) -> Tuple[int, int]:
        """시간프레임별 Stochastic 기간"""
        periods_map = {
            '1h': (14, 3),
            '4h': (14, 3),
            '1d': (14, 3),
            '1w': (14, 3),
            '1M': (14, 3)
        }
        return periods_map.get(timeframe, (14, 3))
    
    def get_williams_period(self, timeframe: str) -> int:
        """시간프레임별 Williams %R 기간"""
        periods_map = {
            '1h': 14,
            '4h': 14,
            '1d': 14,
            '1w': 14,
            '1M': 14
        }
        return periods_map.get(timeframe, 14)
    
    def get_atr_period(self, timeframe: str) -> int:
        """시간프레임별 ATR 기간"""
        periods_map = {
            '1h': 14,
            '4h': 14,
            '1d': 14,
            '1w': 14,
            '1M': 14
        }
        return periods_map.get(timeframe, 14)
    
    def get_momentum_period(self, timeframe: str) -> int:
        """시간프레임별 모멘텀 기간"""
        periods_map = {
            '1h': 10,
            '4h': 10,
            '1d': 10,
            '1w': 10,
            '1M': 10
        }
        return periods_map.get(timeframe, 10)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R 계산"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX 계산"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr = self.calculate_atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """피봇 포인트 계산"""
        pivot = (high + low + close) / 3
        s1 = (pivot * 2) - high
        s2 = pivot - (high - low)
        r1 = (pivot * 2) - low
        r2 = pivot + (high - low)
        return pivot, s1, s2, r1, r2
    
    def analyze_trend(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """추세 분석"""
        if df.empty or len(df) < 50:
            return {'trend': 'neutral', 'strength': 0.0}
        
        latest = df.iloc[-1]
        
        # 이동평균 기반 추세 분석
        ma_signals = []
        ma_periods = self.get_ma_periods(timeframe)
        
        for period in ma_periods:
            if f'ma_{period}' in df.columns:
                ma_value = latest[f'ma_{period}']
                if pd.notna(ma_value):
                    if latest['close'] > ma_value:
                        ma_signals.append(1)
                    elif latest['close'] < ma_value:
                        ma_signals.append(-1)
                    else:
                        ma_signals.append(0)
        
        # MACD 기반 추세 분석
        macd_signal = 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if latest['macd'] > latest['macd_signal']:
                macd_signal = 1
            elif latest['macd'] < latest['macd_signal']:
                macd_signal = -1
        
        # ADX 기반 추세 강도
        trend_strength = 0.0
        if 'adx' in df.columns and pd.notna(latest['adx']):
            trend_strength = latest['adx'] / 100
        
        # 전체 추세 점수 계산
        total_signals = ma_signals + [macd_signal]
        trend_score = sum(total_signals) / len(total_signals) if total_signals else 0
        
        # 추세 방향 결정
        if trend_score > 0.3:
            trend = 'bullish'
        elif trend_score < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'trend': trend,
            'strength': trend_strength,
            'score': trend_score,
            'ma_signals': ma_signals,
            'macd_signal': macd_signal
        }
    
    def analyze_momentum(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """모멘텀 분석"""
        if df.empty or len(df) < 20:
            return {'momentum': 'neutral', 'strength': 0.0}
        
        latest = df.iloc[-1]
        
        # RSI 기반 모멘텀
        rsi_momentum = 0
        if 'rsi' in df.columns and pd.notna(latest['rsi']):
            rsi_value = latest['rsi']
            if rsi_value > 70:
                rsi_momentum = -1  # 과매수
            elif rsi_value < 30:
                rsi_momentum = 1   # 과매도
            else:
                rsi_momentum = (rsi_value - 50) / 50  # 중립 기준 정규화
        
        # Stochastic 기반 모멘텀
        stoch_momentum = 0
        if 'stoch_k' in df.columns and pd.notna(latest['stoch_k']):
            stoch_value = latest['stoch_k']
            if stoch_value > 80:
                stoch_momentum = -1  # 과매수
            elif stoch_value < 20:
                stoch_momentum = 1   # 과매도
            else:
                stoch_momentum = (stoch_value - 50) / 50
        
        # Williams %R 기반 모멘텀
        williams_momentum = 0
        if 'williams_r' in df.columns and pd.notna(latest['williams_r']):
            williams_value = latest['williams_r']
            if williams_value > -20:
                williams_momentum = -1  # 과매수
            elif williams_value < -80:
                williams_momentum = 1   # 과매도
            else:
                williams_momentum = (williams_value + 50) / 50
        
        # 가격 모멘텀
        price_momentum = 0
        if 'momentum' in df.columns and pd.notna(latest['momentum']):
            price_momentum = latest['momentum']
        
        # 전체 모멘텀 점수
        momentum_signals = [rsi_momentum, stoch_momentum, williams_momentum]
        momentum_score = sum(momentum_signals) / len(momentum_signals) if momentum_signals else 0
        
        # 모멘텀 방향 결정
        if momentum_score > 0.3:
            momentum = 'bullish'
        elif momentum_score < -0.3:
            momentum = 'bearish'
        else:
            momentum = 'neutral'
        
        return {
            'momentum': momentum,
            'strength': abs(momentum_score),
            'score': momentum_score,
            'rsi_momentum': rsi_momentum,
            'stoch_momentum': stoch_momentum,
            'williams_momentum': williams_momentum,
            'price_momentum': price_momentum
        }
    
    def analyze_volatility(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """변동성 분석"""
        if df.empty or len(df) < 20:
            return {'volatility': 'normal', 'level': 0.0}
        
        latest = df.iloc[-1]
        
        # ATR 기반 변동성
        atr_volatility = 0.0
        if 'atr' in df.columns and pd.notna(latest['atr']):
            atr_volatility = latest['atr'] / latest['close']
        
        # 볼린저 밴드 기반 변동성
        bb_volatility = 0.0
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
            bb_volatility = bb_width
        
        # 가격 변동성
        price_volatility = 0.0
        if len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                price_volatility = returns.rolling(window=20).std().iloc[-1]
        
        # 평균 변동성
        avg_volatility = np.mean([atr_volatility, bb_volatility, price_volatility])
        
        # 변동성 수준 결정
        if avg_volatility > 0.05:
            volatility_level = 'high'
        elif avg_volatility < 0.02:
            volatility_level = 'low'
        else:
            volatility_level = 'normal'
        
        return {
            'volatility': volatility_level,
            'level': avg_volatility,
            'atr_volatility': atr_volatility,
            'bb_volatility': bb_volatility,
            'price_volatility': price_volatility
        }
    
    def calculate_support_resistance(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """지지/저항 레벨 계산"""
        if df.empty or len(df) < 50:
            return {'support': [], 'resistance': []}
        
        # 피봇 포인트 기반 지지/저항
        support_levels = []
        resistance_levels = []
        
        if all(col in df.columns for col in ['pivot', 's1', 's2', 'r1', 'r2']):
            latest = df.iloc[-1]
            support_levels = [latest['s1'], latest['s2']]
            resistance_levels = [latest['r1'], latest['r2']]
        
        # 과거 고점/저점 기반 지지/저항
        lookback_period = min(100, len(df))
        recent_data = df.tail(lookback_period)
        
        # 로컬 고점/저점 찾기
        highs = recent_data['high']
        lows = recent_data['low']
        
        # 고점 저점 클러스터링
        high_levels = self.find_price_clusters(highs.values, tolerance=0.02)
        low_levels = self.find_price_clusters(lows.values, tolerance=0.02)
        
        resistance_levels.extend(high_levels)
        support_levels.extend(low_levels)
        
        # 중복 제거 및 정렬
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)))
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def find_price_clusters(self, prices: np.ndarray, tolerance: float = 0.02) -> List[float]:
        """가격 클러스터 찾기"""
        if len(prices) < 3:
            return []
        
        clusters = []
        sorted_prices = np.sort(prices)
        
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 2:  # 최소 2개 이상의 가격이 클러스터를 형성
                    clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        # 마지막 클러스터 처리
        if len(current_cluster) >= 2:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def generate_multi_timeframe_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame = None, df_1d: pd.DataFrame = None) -> Dict:
        """다중 시간프레임 신호 생성"""
        signals = {}
        
        # 1시간봉 분석
        if not df_1h.empty:
            df_1h_with_indicators = self.calculate_technical_indicators(df_1h, '1h')
            signals['1h'] = {
                'trend': self.analyze_trend(df_1h_with_indicators, '1h'),
                'momentum': self.analyze_momentum(df_1h_with_indicators, '1h'),
                'volatility': self.analyze_volatility(df_1h_with_indicators, '1h'),
                'support_resistance': self.calculate_support_resistance(df_1h_with_indicators, '1h')
            }
        
        # 4시간봉 분석
        if df_4h is None and not df_1h.empty:
            df_4h = self.resample_data(df_1h, '4h')
        
        if df_4h is not None and not df_4h.empty:
            df_4h_with_indicators = self.calculate_technical_indicators(df_4h, '4h')
            signals['4h'] = {
                'trend': self.analyze_trend(df_4h_with_indicators, '4h'),
                'momentum': self.analyze_momentum(df_4h_with_indicators, '4h'),
                'volatility': self.analyze_volatility(df_4h_with_indicators, '4h'),
                'support_resistance': self.calculate_support_resistance(df_4h_with_indicators, '4h')
            }
        
        # 1일봉 분석
        if df_1d is None and not df_1h.empty:
            df_1d = self.resample_data(df_1h, '1d')
        
        if df_1d is not None and not df_1d.empty:
            df_1d_with_indicators = self.calculate_technical_indicators(df_1d, '1d')
            signals['1d'] = {
                'trend': self.analyze_trend(df_1d_with_indicators, '1d'),
                'momentum': self.analyze_momentum(df_1d_with_indicators, '1d'),
                'volatility': self.analyze_volatility(df_1d_with_indicators, '1d'),
                'support_resistance': self.calculate_support_resistance(df_1d_with_indicators, '1d')
            }
        
        # 통합 신호 계산
        integrated_signal = self.calculate_integrated_signal(signals)
        
        return {
            'individual_signals': signals,
            'integrated_signal': integrated_signal
        }
    
    def calculate_integrated_signal(self, signals: Dict) -> Dict:
        """통합 신호 계산"""
        if not signals:
            return {'signal': 'neutral', 'strength': 0.0, 'confidence': 0.0}
        
        # 각 시간프레임별 신호 점수 계산
        timeframe_scores = {}
        
        for timeframe, signal_data in signals.items():
            score = 0.0
            
            # 추세 점수
            trend_data = signal_data.get('trend', {})
            if trend_data.get('trend') == 'bullish':
                score += trend_data.get('strength', 0) * 0.4
            elif trend_data.get('trend') == 'bearish':
                score -= trend_data.get('strength', 0) * 0.4
            
            # 모멘텀 점수
            momentum_data = signal_data.get('momentum', {})
            if momentum_data.get('momentum') == 'bullish':
                score += momentum_data.get('strength', 0) * 0.3
            elif momentum_data.get('momentum') == 'bearish':
                score -= momentum_data.get('strength', 0) * 0.3
            
            # 변동성 점수 (높은 변동성은 신호 강도를 감소시킴)
            volatility_data = signal_data.get('volatility', {})
            volatility_level = volatility_data.get('level', 0)
            if volatility_level > 0.05:
                score *= 0.8  # 높은 변동성 시 신호 강도 감소
            
            timeframe_scores[timeframe] = score
        
        # 가중평균 계산
        weighted_score = sum(
            timeframe_scores.get(tf, 0) * self.weights.get(tf, 0)
            for tf in self.timeframes
        )
        
        # 신호 방향 결정
        if weighted_score > 0.3:
            signal = 'bullish'
        elif weighted_score < -0.3:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        # 신호 강도 및 신뢰도 계산
        strength = abs(weighted_score)
        confidence = min(1.0, strength * 2)  # 신뢰도는 강도의 2배 (최대 1.0)
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'timeframe_scores': timeframe_scores
        }
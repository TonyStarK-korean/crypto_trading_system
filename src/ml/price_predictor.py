"""
가격 예측 ML 모델
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    """가격 예측 시스템"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링"""
        try:
            features = df.copy()
            
            # 가격 변화율
            features['price_change'] = features['close'].pct_change()
            features['price_change_5'] = features['close'].pct_change(5)
            features['price_change_10'] = features['close'].pct_change(10)
            
            # 이동평균 비율
            features['sma_ratio_20'] = features['close'] / features['sma_20']
            features['sma_ratio_50'] = features['close'] / features['sma_50']
            features['ema_ratio_12'] = features['close'] / features['ema_12']
            features['ema_ratio_26'] = features['close'] / features['ema_26']
            
            # 볼린저 밴드 위치
            features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            
            # RSI 변형
            features['rsi_ma'] = features['rsi'].rolling(window=10).mean()
            features['rsi_std'] = features['rsi'].rolling(window=10).std()
            
            # MACD 신호
            features['macd_signal'] = np.where(features['macd'] > features['macd_signal'], 1, -1)
            features['macd_histogram_change'] = features['macd_histogram'].diff()
            
            # 거래량 지표
            features['volume_ma_ratio'] = features['volume'] / features['volume_sma']
            features['volume_price_trend'] = features['volume'] * features['price_change']
            
            # 변동성 지표
            features['volatility'] = features['close'].rolling(window=20).std()
            features['volatility_ratio'] = features['volatility'] / features['close']
            
            # 시간 특성
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            
            # 지연 특성 (lag features)
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = features['close'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
                features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            
            # 이동평균 교차
            features['sma_cross'] = np.where(features['sma_20'] > features['sma_50'], 1, -1)
            features['ema_cross'] = np.where(features['ema_12'] > features['ema_26'], 1, -1)
            
            # 목표 변수 생성 (다음 기간 수익률)
            features['target_return_1'] = features['close'].shift(-1) / features['close'] - 1
            features['target_return_5'] = features['close'].shift(-5) / features['close'] - 1
            features['target_direction_1'] = np.where(features['target_return_1'] > 0, 1, 0)
            
            # NaN 값 처리
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"특성 엔지니어링 오류: {e}")
            return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """중요한 특성 선택"""
        feature_candidates = [
            'price_change', 'price_change_5', 'price_change_10',
            'sma_ratio_20', 'sma_ratio_50', 'ema_ratio_12', 'ema_ratio_26',
            'bb_position', 'bb_width', 'rsi', 'rsi_ma', 'rsi_std',
            'macd', 'macd_signal', 'macd_histogram_change',
            'volume_ma_ratio', 'volume_price_trend',
            'volatility', 'volatility_ratio',
            'hour', 'day_of_week', 'month',
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'volume_lag_1', 'volume_lag_2',
            'rsi_lag_1', 'rsi_lag_2',
            'sma_cross', 'ema_cross'
        ]
        
        available_features = [col for col in feature_candidates if col in df.columns]
        return available_features
    
    def train_model(self, df: pd.DataFrame, target_col: str = 'target_return_1') -> bool:
        """모델 훈련"""
        try:
            # 특성 준비
            features_df = self.prepare_features(df)
            self.feature_columns = self.select_features(features_df)
            self.target_columns = [target_col]
            
            if len(self.feature_columns) == 0:
                logger.error("사용 가능한 특성이 없습니다")
                return False
            
            # 데이터 분할
            X = features_df[self.feature_columns]
            y = features_df[target_col]
            
            # NaN 값 제거
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 100:
                logger.error("훈련 데이터가 부족합니다")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target_col] = scaler
            
            # 모델 선택 및 훈련
            if self.model_type == "ensemble":
                self.models[target_col] = self._train_ensemble_model(X_train_scaled, y_train)
            elif self.model_type == "random_forest":
                self.models[target_col] = RandomForestRegressor(n_estimators=100, random_state=42)
                self.models[target_col].fit(X_train_scaled, y_train)
            elif self.model_type == "gradient_boosting":
                self.models[target_col] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.models[target_col].fit(X_train_scaled, y_train)
            else:
                self.models[target_col] = LinearRegression()
                self.models[target_col].fit(X_train_scaled, y_train)
            
            # 모델 평가
            y_pred = self.models[target_col].predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"모델 훈련 완료 - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"모델 훈련 오류: {e}")
            return False
    
    def _train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """앙상블 모델 훈련"""
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            LinearRegression()
        ]
        
        # 각 모델 훈련
        trained_models = []
        for model in models:
            model.fit(X_train, y_train)
            trained_models.append(model)
        
        return trained_models
    
    def predict(self, df: pd.DataFrame, target_col: str = 'target_return_1') -> Optional[float]:
        """가격 예측"""
        try:
            if not self.is_trained or target_col not in self.models:
                logger.error("모델이 훈련되지 않았습니다")
                return None
            
            # 특성 준비
            features_df = self.prepare_features(df)
            
            if len(features_df) == 0:
                return None
            
            # 최신 데이터 사용
            latest_features = features_df[self.feature_columns].iloc[-1:]
            
            if latest_features.isna().any().any():
                logger.warning("예측을 위한 특성에 NaN 값이 있습니다")
                return None
            
            # 스케일링
            scaler = self.scalers[target_col]
            features_scaled = scaler.transform(latest_features)
            
            # 예측
            model = self.models[target_col]
            
            if isinstance(model, list):  # 앙상블 모델
                predictions = [m.predict(features_scaled)[0] for m in model]
                prediction = np.mean(predictions)
            else:
                prediction = model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"예측 오류: {e}")
            return None
    
    def predict_probability(self, df: pd.DataFrame, target_col: str = 'target_direction_1') -> Optional[float]:
        """방향 예측 확률"""
        try:
            # 수익률 예측
            return_pred = self.predict(df, 'target_return_1')
            
            if return_pred is None:
                return None
            
            # 확률로 변환 (간단한 방법)
            # 실제로는 분류 모델을 별도로 훈련하는 것이 좋습니다
            probability = 1 / (1 + np.exp(-return_pred * 10))  # 시그모이드 변환
            
            return probability
            
        except Exception as e:
            logger.error(f"확률 예측 오류: {e}")
            return None
    
    def save_model(self, filepath: str):
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"모델 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"모델 저장 오류: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """모델 로드"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"모델 파일을 찾을 수 없습니다: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data['target_columns']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            return False
    
    def get_feature_importance(self, target_col: str = 'target_return_1') -> Dict[str, float]:
        """특성 중요도 반환"""
        try:
            if target_col not in self.models:
                return {}
            
            model = self.models[target_col]
            
            if isinstance(model, list):
                # 앙상블 모델의 경우 첫 번째 모델 사용
                model = model[0]
            
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                logger.warning("이 모델은 특성 중요도를 지원하지 않습니다")
                return {}
                
        except Exception as e:
            logger.error(f"특성 중요도 계산 오류: {e}")
            return {} 
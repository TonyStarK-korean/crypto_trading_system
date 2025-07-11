"""
머신러닝 기반 예측 모델
다양한 ML 모델을 사용한 가격 예측 및 신호 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn을 사용할 수 없습니다. 기본 예측 모델을 사용합니다.")

logger = logging.getLogger(__name__)

class MLPredictor:
    """머신러닝 예측 모델"""
    
    def __init__(self,
                 model_type: str = 'ensemble',  # 'ensemble', 'classification', 'regression'
                 prediction_horizon: int = 1,    # 예측 시점 (1시간 후)
                 feature_window: int = 50,       # 특성 계산 윈도우
                 min_data_points: int = 200,     # 최소 데이터 포인트
                 retrain_frequency: int = 100):  # 재훈련 주기
        
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.min_data_points = min_data_points
        self.retrain_frequency = retrain_frequency
        
        # 모델 저장소
        self.models = {}
        self.scalers = {}
        self.is_trained = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        # 훈련 카운터
        self.training_count = 0
        
        if SKLEARN_AVAILABLE:
            self.initialize_models()
        else:
            self.models = {'dummy': None}
            self.is_trained = {'dummy': False}
    
    def initialize_models(self):
        """모델 초기화"""
        if not SKLEARN_AVAILABLE:
            return
        
        # 분류 모델들 (방향 예측)
        classification_models = {
            'rf_classifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb_classifier': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm_classifier': SVC(
                probability=True,
                random_state=42
            ),
            'naive_bayes': GaussianNB(),
            'knn_classifier': KNeighborsClassifier(n_neighbors=5)
        }
        
        # 회귀 모델들 (가격 예측)
        regression_models = {
            'rf_regressor': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'linear_regression': LinearRegression(),
            'svm_regressor': SVR()
        }
        
        # 모델 타입에 따라 초기화
        if self.model_type == 'classification':
            self.models = classification_models
        elif self.model_type == 'regression':
            self.models = regression_models
        else:  # ensemble
            self.models = {**classification_models, **regression_models}
        
        # 스케일러 초기화
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # 훈련 상태 초기화
        self.is_trained = {model_name: False for model_name in self.models.keys()}
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 추출"""
        if df.empty or len(df) < self.feature_window:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(index=df.index)
        
        # 가격 기반 특성
        features_df['price'] = df['close']
        features_df['returns_1h'] = df['close'].pct_change(1)
        features_df['returns_4h'] = df['close'].pct_change(4)
        features_df['returns_24h'] = df['close'].pct_change(24)
        features_df['returns_7d'] = df['close'].pct_change(168)  # 7일 (168시간)
        
        # 로그 수익률
        features_df['log_returns_1h'] = np.log(df['close'] / df['close'].shift(1))
        features_df['log_returns_4h'] = np.log(df['close'] / df['close'].shift(4))
        
        # 가격 변동성
        for window in [6, 12, 24, 48]:
            features_df[f'volatility_{window}h'] = features_df['returns_1h'].rolling(window).std()
        
        # 이동평균 관련 특성
        for period in [5, 10, 20, 50, 100]:
            ma = df['close'].rolling(period).mean()
            features_df[f'ma_{period}'] = ma
            features_df[f'price_to_ma_{period}'] = df['close'] / ma
            features_df[f'ma_{period}_slope'] = ma.diff(5) / ma.shift(5)
        
        # 기술적 지표
        features_df = self.add_technical_indicators(features_df, df)
        
        # 거래량 특성
        features_df = self.add_volume_features(features_df, df)
        
        # 시간 기반 특성
        features_df = self.add_time_features(features_df, df)
        
        # 고차 특성 (상호작용)
        features_df = self.add_interaction_features(features_df)
        
        # 결측값 처리
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def add_technical_indicators(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        # RSI
        features_df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        macd, signal, histogram = self.calculate_macd(df['close'])
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_histogram'] = histogram
        
        # 볼린저 밴드
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        features_df['bb_upper'] = bb_upper
        features_df['bb_middle'] = bb_middle
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        if all(col in df.columns for col in ['high', 'low']):
            stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
            features_df['stoch_k'] = stoch_k
            features_df['stoch_d'] = stoch_d
        
        # Williams %R
        if all(col in df.columns for col in ['high', 'low']):
            features_df['williams_r'] = self.calculate_williams_r(df['high'], df['low'], df['close'])
        
        # ATR
        if all(col in df.columns for col in ['high', 'low']):
            features_df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        return features_df
    
    def add_volume_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 특성 추가"""
        if 'volume' not in df.columns:
            return features_df
        
        # 거래량 이동평균
        for period in [5, 10, 20]:
            vol_ma = df['volume'].rolling(period).mean()
            features_df[f'volume_ma_{period}'] = vol_ma
            features_df[f'volume_ratio_{period}'] = df['volume'] / vol_ma
        
        # 거래량 변화율
        features_df['volume_change'] = df['volume'].pct_change()
        
        # 가격-거래량 관계
        features_df['price_volume_trend'] = features_df['returns_1h'] * features_df['volume_change']
        
        # OBV (On-Balance Volume)
        obv = []
        prev_obv = 0
        for i, (_, row) in enumerate(df.iterrows()):
            if i == 0:
                obv.append(row['volume'])
                prev_obv = row['volume']
            else:
                prev_close = df['close'].iloc[i-1]
                if row['close'] > prev_close:
                    current_obv = prev_obv + row['volume']
                elif row['close'] < prev_close:
                    current_obv = prev_obv - row['volume']
                else:
                    current_obv = prev_obv
                obv.append(current_obv)
                prev_obv = current_obv
        
        features_df['obv'] = obv
        features_df['obv_ma'] = features_df['obv'].rolling(20).mean()
        
        return features_df
    
    def add_time_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """시간 기반 특성 추가"""
        # 시간 특성
        features_df['hour'] = df.index.hour
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['month'] = df.index.month
        
        # 순환 시간 특성 (sin/cos 변환)
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        # 시장 세션 (UTC 기준)
        features_df['asian_session'] = ((features_df['hour'] >= 0) & (features_df['hour'] < 8)).astype(int)
        features_df['european_session'] = ((features_df['hour'] >= 8) & (features_df['hour'] < 16)).astype(int)
        features_df['american_session'] = ((features_df['hour'] >= 16) & (features_df['hour'] < 24)).astype(int)
        
        # 주말 여부
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        return features_df
    
    def add_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """상호작용 특성 추가"""
        # 가격과 거래량 상호작용
        if 'price_to_ma_20' in features_df.columns and 'volume_ratio_20' in features_df.columns:
            features_df['price_volume_interaction'] = features_df['price_to_ma_20'] * features_df['volume_ratio_20']
        
        # RSI와 변동성 상호작용
        if 'rsi' in features_df.columns and 'volatility_24h' in features_df.columns:
            features_df['rsi_volatility_interaction'] = features_df['rsi'] * features_df['volatility_24h']
        
        # MACD와 볼린저 밴드 상호작용
        if 'macd_histogram' in features_df.columns and 'bb_position' in features_df.columns:
            features_df['macd_bb_interaction'] = features_df['macd_histogram'] * features_df['bb_position']
        
        return features_df
    
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
    
    def prepare_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """타겟 변수 준비"""
        targets = {}
        
        # 분류 타겟 (방향 예측)
        future_returns = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # 이진 분류: 상승/하락
        targets['binary_direction'] = (future_returns > 0).astype(int)
        
        # 다중 분류: 강한 상승/상승/중립/하락/강한 하락
        targets['multi_direction'] = pd.cut(
            future_returns,
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=[0, 1, 2, 3, 4]  # 강한 하락, 하락, 중립, 상승, 강한 상승
        ).astype(int)
        
        # 회귀 타겟 (가격 예측)
        targets['future_price'] = df['close'].shift(-self.prediction_horizon)
        targets['future_returns'] = future_returns
        
        return targets
    
    def train_models(self, df: pd.DataFrame):
        """모델 훈련"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn을 사용할 수 없어 모델 훈련을 건너뜁니다.")
            return
        
        if len(df) < self.min_data_points:
            logger.info(f"데이터가 부족합니다. 필요: {self.min_data_points}, 현재: {len(df)}")
            return
        
        logger.info("ML 모델 훈련 시작...")
        
        # 특성 추출
        features_df = self.extract_features(df)
        if features_df.empty:
            return
        
        # 타겟 준비
        targets = self.prepare_targets(df)
        
        # 유효한 데이터만 사용
        valid_idx = features_df.index.intersection(list(targets.values())[0].index)
        valid_idx = valid_idx[~pd.isna(list(targets.values())[0][valid_idx])]
        
        if len(valid_idx) < 50:
            logger.warning("유효한 데이터가 부족합니다.")
            return
        
        features_df = features_df.loc[valid_idx]
        for key in targets:
            targets[key] = targets[key].loc[valid_idx]
        
        # 무한값 및 결측값 처리
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 특성 선택 (분산이 0인 특성 제거)
        feature_variance = features_df.var()
        valid_features = feature_variance[feature_variance > 1e-8].index
        features_df = features_df[valid_features]
        
        X = features_df.values
        
        # 데이터 스케일링
        for scaler_name, scaler in self.scalers.items():
            try:
                scaler.fit(X)
            except Exception as e:
                logger.error(f"스케일러 {scaler_name} 훈련 실패: {e}")
        
        # 각 모델별 훈련
        for model_name, model in self.models.items():
            try:
                if model is None:
                    continue
                
                # 모델 타입에 따른 타겟 선택
                if 'classifier' in model_name or model_name in ['logistic', 'naive_bayes', 'knn_classifier']:
                    y = targets['binary_direction'].values
                    task_type = 'classification'
                else:
                    y = targets['future_returns'].values
                    task_type = 'regression'
                
                # 데이터 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
                )
                
                # 스케일링 적용
                scaler = self.scalers['standard']
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 모델 훈련
                model.fit(X_train_scaled, y_train)
                
                # 성능 평가
                y_pred = model.predict(X_test_scaled)
                
                if task_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    self.model_performance[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'task_type': task_type
                    }
                    
                    logger.info(f"{model_name} - 정확도: {accuracy:.3f}, F1: {f1:.3f}")
                    
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.model_performance[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2,
                        'task_type': task_type
                    }
                    
                    logger.info(f"{model_name} - MSE: {mse:.6f}, R2: {r2:.3f}")
                
                # 특성 중요도 저장 (지원하는 모델만)
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(valid_features, model.feature_importances_))
                    self.feature_importance[model_name] = importance_dict
                
                self.is_trained[model_name] = True
                
            except Exception as e:
                logger.error(f"모델 {model_name} 훈련 실패: {e}")
                self.is_trained[model_name] = False
        
        self.training_count += 1
        logger.info("ML 모델 훈련 완료")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """예측 수행"""
        if not SKLEARN_AVAILABLE:
            return {'ensemble_probability': 0.5, 'ensemble_direction': 'neutral'}
        
        # 특성 추출
        features_df = self.extract_features(df)
        if features_df.empty:
            return {'ensemble_probability': 0.5, 'ensemble_direction': 'neutral'}
        
        # 최신 데이터 사용
        latest_features = features_df.iloc[-1:].values
        
        # 무한값 및 결측값 처리
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = {}
        probabilities = []
        
        # 각 모델별 예측
        for model_name, model in self.models.items():
            if not self.is_trained.get(model_name, False) or model is None:
                continue
            
            try:
                # 스케일링 적용
                scaler = self.scalers['standard']
                scaled_features = scaler.transform(latest_features)
                
                # 예측 수행
                if hasattr(model, 'predict_proba'):  # 분류 모델
                    proba = model.predict_proba(scaled_features)[0]
                    predictions[model_name] = {
                        'prediction': model.predict(scaled_features)[0],
                        'probability': proba[1] if len(proba) > 1 else 0.5  # 상승 확률
                    }
                    probabilities.append(proba[1] if len(proba) > 1 else 0.5)
                else:  # 회귀 모델
                    pred_value = model.predict(scaled_features)[0]
                    predictions[model_name] = {
                        'prediction': pred_value,
                        'probability': 0.5 + pred_value  # 수익률을 확률로 변환
                    }
                    probabilities.append(0.5 + pred_value)
                
            except Exception as e:
                logger.error(f"모델 {model_name} 예측 실패: {e}")
        
        # 앙상블 예측
        if probabilities:
            ensemble_probability = np.mean(probabilities)
            ensemble_probability = np.clip(ensemble_probability, 0, 1)
            
            if ensemble_probability > 0.6:
                ensemble_direction = 'bullish'
            elif ensemble_probability < 0.4:
                ensemble_direction = 'bearish'
            else:
                ensemble_direction = 'neutral'
        else:
            ensemble_probability = 0.5
            ensemble_direction = 'neutral'
        
        return {
            'individual_predictions': predictions,
            'ensemble_probability': ensemble_probability,
            'ensemble_direction': ensemble_direction,
            'model_count': len(predictions),
            'confidence': abs(ensemble_probability - 0.5) * 2  # 0-1 범위의 신뢰도
        }
    
    def should_retrain(self) -> bool:
        """재훈련 필요 여부 확인"""
        return self.training_count % self.retrain_frequency == 0
    
    def save_models(self, filepath: str):
        """모델 저장"""
        if not SKLEARN_AVAILABLE:
            return
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'is_trained': self.is_trained,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'training_count': self.training_count
        }
        
        try:
            joblib.dump(model_data, filepath)
            logger.info(f"모델이 저장되었습니다: {filepath}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def load_models(self, filepath: str):
        """모델 로드"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.is_trained = model_data['is_trained']
            self.model_performance = model_data['model_performance']
            self.feature_importance = model_data['feature_importance']
            self.training_count = model_data['training_count']
            logger.info(f"모델이 로드되었습니다: {filepath}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    def get_feature_importance_report(self) -> Dict:
        """특성 중요도 보고서 생성"""
        if not self.feature_importance:
            return {}
        
        # 모든 모델의 특성 중요도 평균 계산
        all_features = set()
        for importance_dict in self.feature_importance.values():
            all_features.update(importance_dict.keys())
        
        avg_importance = {}
        for feature in all_features:
            importances = []
            for importance_dict in self.feature_importance.values():
                if feature in importance_dict:
                    importances.append(importance_dict[feature])
            
            if importances:
                avg_importance[feature] = np.mean(importances)
        
        # 중요도 순으로 정렬
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_features': sorted_features[:20],  # 상위 20개 특성
            'individual_model_importance': self.feature_importance,
            'total_features': len(all_features)
        }
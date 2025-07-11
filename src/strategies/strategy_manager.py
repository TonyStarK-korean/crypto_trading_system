"""
전략 관리자
설정 파일을 읽어서 전략을 초기화하고 관리
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .advanced_strategy import AdvancedIntegratedStrategy
from ..core.risk_manager import RiskManager, PortfolioManager
from ..core.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..ml.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)

class StrategyManager:
    """전략 관리자"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/advanced_strategy_config.yaml"
        self.config = None
        self.strategy = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.mtf_analyzer = None
        self.ml_predictor = None
        
        # 설정 로드
        self.load_config()
        
        # 컴포넌트 초기화
        self.initialize_components()
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            config_file_path = Path(self.config_path)
            if not config_file_path.exists():
                # 프로젝트 루트에서 찾기
                project_root = Path(__file__).parent.parent.parent
                config_file_path = project_root / self.config_path
            
            if not config_file_path.exists():
                logger.error(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
                self.config = self.get_default_config()
                return
            
            with open(config_file_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            logger.info(f"설정 파일 로드 완료: {config_file_path}")
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy': {
                'name': 'advanced_integrated_strategy',
                'version': '1.0.0'
            },
            'adaptive': {
                'enabled': True,
                'adaptive_window': 50,
                'volatility_lookback': 20
            },
            'multi_timeframe': {
                'enabled': True,
                'timeframes': ['1h', '4h', '1d'],
                'weights': {'1h': 0.5, '4h': 0.3, '1d': 0.2}
            },
            'machine_learning': {
                'enabled': True,
                'model_type': 'ensemble',
                'min_data_points': 200
            },
            'risk_parity': {
                'enabled': True,
                'max_portfolio_risk': 0.02,
                'max_position_size': 0.1
            },
            'entry_conditions': {
                'mtf_signal_threshold': 0.7,
                'ml_signal_threshold': 0.7,
                'signal_weights': {
                    'multi_timeframe': 0.6,
                    'machine_learning': 0.4
                }
            },
            'exit_conditions': {
                'dynamic_levels': True,
                'base_stop_loss': 0.02,
                'base_take_profit': 0.05
            },
            'backtest': {
                'initial_balance': 10000.0,
                'commission': 0.001
            }
        }
    
    def initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 리스크 관리자 초기화
            if self.config.get('risk_parity', {}).get('enabled', True):
                risk_config = self.config.get('risk_parity', {})
                self.risk_manager = RiskManager(
                    max_portfolio_risk=risk_config.get('max_portfolio_risk', 0.02),
                    max_position_size=risk_config.get('max_position_size', 0.1),
                    max_correlation=risk_config.get('max_correlation', 0.7),
                    volatility_lookback=risk_config.get('volatility_lookback', 20),
                    var_confidence=risk_config.get('var_confidence', 0.05)
                )
                
                self.portfolio_manager = PortfolioManager(self.risk_manager)
            
            # 다중 시간프레임 분석기 초기화
            if self.config.get('multi_timeframe', {}).get('enabled', True):
                mtf_config = self.config.get('multi_timeframe', {})
                self.mtf_analyzer = MultiTimeframeAnalyzer(
                    timeframes=mtf_config.get('timeframes', ['1h', '4h', '1d']),
                    weights=mtf_config.get('weights', {'1h': 0.5, '4h': 0.3, '1d': 0.2})
                )
            
            # 머신러닝 예측기 초기화
            if self.config.get('machine_learning', {}).get('enabled', True):
                ml_config = self.config.get('machine_learning', {})
                self.ml_predictor = MLPredictor(
                    model_type=ml_config.get('model_type', 'ensemble'),
                    prediction_horizon=ml_config.get('prediction_horizon', 1),
                    feature_window=ml_config.get('feature_window', 50),
                    min_data_points=ml_config.get('min_data_points', 200),
                    retrain_frequency=ml_config.get('retrain_frequency', 100)
                )
            
            # 고급 통합 전략 초기화
            adaptive_config = self.config.get('adaptive', {})
            mtf_config = self.config.get('multi_timeframe', {})
            ml_config = self.config.get('machine_learning', {})
            risk_config = self.config.get('risk_parity', {})
            
            self.strategy = AdvancedIntegratedStrategy(
                # 적응형 파라미터
                adaptive_window=adaptive_config.get('adaptive_window', 50),
                volatility_lookback=adaptive_config.get('volatility_lookback', 20),
                
                # 다중 시간프레임 파라미터
                timeframes=mtf_config.get('timeframes', ['1h', '4h', '1d']),
                
                # 머신러닝 파라미터
                ml_enabled=ml_config.get('enabled', True),
                ml_lookback=ml_config.get('min_data_points', 200),
                ml_features=ml_config.get('feature_window', 50),
                
                # 리스크 패리티 파라미터
                risk_target=risk_config.get('risk_per_trade', 0.02),
                max_position_size=risk_config.get('max_position_size', 0.1),
                volatility_target=risk_config.get('volatility_target', 0.15)
            )
            
            logger.info("모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def get_strategy(self) -> AdvancedIntegratedStrategy:
        """전략 인스턴스 반환"""
        return self.strategy
    
    def get_risk_manager(self) -> Optional[RiskManager]:
        """리스크 관리자 반환"""
        return self.risk_manager
    
    def get_portfolio_manager(self) -> Optional[PortfolioManager]:
        """포트폴리오 관리자 반환"""
        return self.portfolio_manager
    
    def get_mtf_analyzer(self) -> Optional[MultiTimeframeAnalyzer]:
        """다중 시간프레임 분석기 반환"""
        return self.mtf_analyzer
    
    def get_ml_predictor(self) -> Optional[MLPredictor]:
        """머신러닝 예측기 반환"""
        return self.ml_predictor
    
    def get_config(self) -> Dict[str, Any]:
        """설정 반환"""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        self.config.update(new_config)
        self.initialize_components()
        logger.info("설정 업데이트 및 컴포넌트 재초기화 완료")
    
    def save_config(self, filepath: str = None):
        """설정 저장"""
        if filepath is None:
            filepath = self.config_path
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
            logger.info(f"설정이 저장되었습니다: {filepath}")
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        try:
            # 필수 섹션 확인
            required_sections = ['strategy', 'adaptive', 'multi_timeframe', 'machine_learning', 'risk_parity']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"필수 설정 섹션이 누락되었습니다: {section}")
                    return False
            
            # 시간프레임 설정 확인
            mtf_config = self.config.get('multi_timeframe', {})
            timeframes = mtf_config.get('timeframes', [])
            weights = mtf_config.get('weights', {})
            
            if not timeframes:
                logger.error("시간프레임이 설정되지 않았습니다")
                return False
            
            for tf in timeframes:
                if tf not in weights:
                    logger.error(f"시간프레임 {tf}의 가중치가 설정되지 않았습니다")
                    return False
            
            # 가중치 합계 확인
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"시간프레임 가중치 합계가 1.0이 아닙니다: {total_weight}")
            
            # 리스크 설정 확인
            risk_config = self.config.get('risk_parity', {})
            max_portfolio_risk = risk_config.get('max_portfolio_risk', 0)
            max_position_size = risk_config.get('max_position_size', 0)
            
            if max_portfolio_risk <= 0 or max_portfolio_risk > 0.1:
                logger.error(f"포트폴리오 최대 리스크가 유효하지 않습니다: {max_portfolio_risk}")
                return False
            
            if max_position_size <= 0 or max_position_size > 1.0:
                logger.error(f"최대 포지션 크기가 유효하지 않습니다: {max_position_size}")
                return False
            
            logger.info("설정 유효성 검사 통과")
            return True
            
        except Exception as e:
            logger.error(f"설정 유효성 검사 실패: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        strategy_config = self.config.get('strategy', {})
        
        enabled_features = []
        if self.config.get('adaptive', {}).get('enabled', False):
            enabled_features.append("적응형 임계값")
        if self.config.get('multi_timeframe', {}).get('enabled', False):
            enabled_features.append("다중 시간프레임")
        if self.config.get('machine_learning', {}).get('enabled', False):
            enabled_features.append("머신러닝")
        if self.config.get('risk_parity', {}).get('enabled', False):
            enabled_features.append("리스크 패리티")
        
        return {
            'name': strategy_config.get('name', 'Unknown'),
            'version': strategy_config.get('version', '1.0.0'),
            'description': strategy_config.get('description', ''),
            'enabled_features': enabled_features,
            'timeframes': self.config.get('multi_timeframe', {}).get('timeframes', []),
            'initial_balance': self.config.get('backtest', {}).get('initial_balance', 10000),
            'max_portfolio_risk': self.config.get('risk_parity', {}).get('max_portfolio_risk', 0.02),
            'commission': self.config.get('backtest', {}).get('commission', 0.001)
        }
    
    def print_strategy_summary(self):
        """전략 요약 출력"""
        info = self.get_strategy_info()
        
        print("="*60)
        print(f"🚀 {info['name']} v{info['version']}")
        print("="*60)
        print(f"설명: {info['description']}")
        print(f"\n활성화된 기능:")
        for feature in info['enabled_features']:
            print(f"  ✓ {feature}")
        
        print(f"\n시간프레임: {', '.join(info['timeframes'])}")
        print(f"초기 자본: {info['initial_balance']:,} USDT")
        print(f"최대 포트폴리오 리스크: {info['max_portfolio_risk']*100:.1f}%")
        print(f"수수료: {info['commission']*100:.2f}%")
        print("="*60)
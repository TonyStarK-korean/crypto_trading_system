#!/usr/bin/env python3
"""
정식 백테스트 시스템
콘솔 기반 메뉴 시스템으로 데이터 다운로드와 백테스트 실행을 제공
"""

import sys
import os
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simple_backtest import SimpleBacktestEngine
from src.strategies.pump_detection_strategy import PumpDetectionStrategy


class BacktestSystem:
    """백테스트 시스템"""
    
    def __init__(self):
        # 바이낸스 스팟 마켓으로 연결 (더 많은 심볼 지원)
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'enableRateLimit': True
        })
        self.data_dir = "data"
        self.results_dir = "results"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 전략 초기화
        self.strategies = {
            '급등 초입 진입전략': PumpDetectionStrategy(),
            '고급 통합전략': None,  # 새로운 전략 (동적 로딩)
            '볼린저 밴드 브레이크아웃': None,  # 나중에 구현
            '이동평균 교차': None  # 나중에 구현
        }
        
    def show_main_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*50)
        print("🚀 암호화폐 백테스트 시스템")
        print("="*50)
        print("1. 백테스트 데이터 다운로드")
        print("2. 백테스트 실행")
        print("3. 종료")
        print("="*50)
        
    def show_download_menu(self):
        """다운로드 메뉴 표시"""
        print("\n" + "="*50)
        print("📥 백테스트 데이터 다운로드")
        print("="*50)
        print("1. 심볼 목록 보기 및 선택")
        print("2. 전체 심볼 다운로드")
        print("3. 이전 메뉴로")
        print("="*50)
        
    def show_strategy_menu(self):
        """전략 선택 메뉴 표시"""
        print("\n" + "="*50)
        print("📊 백테스트 전략 선택")
        print("="*50)
        print("1. 급등 초입 진입전략 (1H봉 기준)")
        print("2. 고급 통합전략 (적응형+다중시간프레임+ML+리스크패리티)")
        print("3. 볼린저 밴드 브레이크아웃 전략")
        print("4. 이동평균 교차 전략")
        print("5. 이전 메뉴로")
        print("="*50)
        
    def get_date_range(self):
        """날짜 범위 입력 받기"""
        print("\n📅 데이터 다운로드 기간 설정")
        print("="*30)
        
        while True:
            try:
                start_date = input("시작 날짜 (YYYY-MM-DD): ").strip()
                end_date = input("종료 날짜 (YYYY-MM-DD): ").strip()
                
                # 날짜 형식 검증
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
                
                if start_date > end_date:
                    print("❌ 시작 날짜가 종료 날짜보다 늦을 수 없습니다.")
                    continue
                    
                return start_date, end_date
                
            except ValueError:
                print("❌ 올바른 날짜 형식을 입력하세요 (YYYY-MM-DD)")
                
    def get_binance_usdtp_symbols(self):
        """바이낸스 스팟 마켓에서 USDT 페어 심볼 목록 가져오기"""
        try:
            print("🔄 바이낸스 스팟 마켓에서 USDT 페어 심볼 목록을 가져오는 중...")
            markets = self.exchange.load_markets()
            usdt_symbols = []
            for symbol, market in markets.items():
                if symbol.endswith('/USDT') and market.get('active'):
                    usdt_symbols.append(symbol)
            usdt_symbols = sorted(list(set(usdt_symbols)))
            print(f"✅ 총 {len(usdt_symbols)}개의 USDT 페어 심볼을 찾았습니다.")
            return usdt_symbols
        except Exception as e:
            print(f"❌ 심볼 목록 가져오기 실패: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    def download_symbol_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '1h'):
        """개별 심볼 데이터 다운로드 (심볼 지원 여부 체크)"""
        try:
            markets = self.exchange.load_markets()
            if symbol not in markets:
                print(f"❌ {symbol}: 바이낸스 스팟 마켓에서 지원되지 않습니다.")
                return None
            print(f"📥 {symbol} 데이터 다운로드 중...")
            
            # 날짜를 timestamp로 변환
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            # 지정된 기간의 데이터 수집
            all_ohlcv = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                try:
                    # 1000개씩 배치로 데이터 수집
                    ohlcv_batch = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_timestamp, 
                        limit=1000
                    )
                    
                    if not ohlcv_batch:
                        break
                    
                    all_ohlcv.extend(ohlcv_batch)
                    
                    # 다음 배치의 시작 시간 설정
                    last_timestamp = ohlcv_batch[-1][0]
                    if last_timestamp <= current_timestamp:
                        break
                    
                    current_timestamp = last_timestamp + 1
                    
                    # API 제한 방지
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️ {symbol} 배치 다운로드 오류: {e}")
                    break
            
            if not all_ohlcv:
                print(f"⚠️ {symbol}: 데이터가 없습니다.")
                return None
            
            # DataFrame으로 변환
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 날짜 범위 필터링 (정확한 기간 적용)
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # 종료일 포함
            df = df[(df.index >= start_date_dt) & (df.index < end_date_dt)]
            
            if len(df) == 0:
                print(f"⚠️ {symbol}: 지정된 기간에 데이터가 없습니다.")
                return None
            
            # 파일명을 다운로드 기간으로 설정
            start_date_str = start_date.replace('-', '')
            end_date_str = end_date.replace('-', '')
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date_str}_{end_date_str}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            
            print(f"✅ {symbol}: {len(df)}개 캔들 저장 완료 - {filepath}")
            print(f"   기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
            return filepath
            
        except Exception as e:
            print(f"❌ {symbol} 다운로드 실패: {e}")
            return None
    
    def download_all_symbols(self, start_date: str, end_date: str, timeframe: str = '1h'):
        """전체 심볼 데이터 다운로드 및 통합"""
        try:
            symbols = self.get_binance_usdtp_symbols()
            
            print(f"\n📊 총 {len(symbols)}개 심볼 다운로드 시작...")
            print("="*50)
            
            downloaded_files = []
            successful_count = 0
            
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/{len(symbols)}] {symbol} 처리 중...")
                
                filepath = self.download_symbol_data(symbol, start_date, end_date, timeframe)
                if filepath:
                    downloaded_files.append(filepath)
                    successful_count += 1
                
                # API 제한 방지
                time.sleep(0.1)
            
            print(f"\n✅ 다운로드 완료: {successful_count}/{len(symbols)} 성공")
            
            if downloaded_files:
                # 파일 통합
                print("\n🔄 파일 통합 중...")
                self.merge_csv_files(downloaded_files, start_date, end_date, timeframe)
                
                # 개별 파일 삭제
                print("🗑️ 개별 파일 삭제 중...")
                for filepath in downloaded_files:
                    try:
                        os.remove(filepath)
                        print(f"삭제: {os.path.basename(filepath)}")
                    except:
                        pass
                
                print("✅ 파일 통합 및 정리 완료!")
            
        except Exception as e:
            print(f"❌ 전체 다운로드 실패: {e}")
    
    def merge_csv_files(self, file_paths: list, start_date: str, end_date: str, timeframe: str):
        """CSV 파일들을 하나로 통합"""
        try:
            all_data = []
            
            for filepath in file_paths:
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    # 심볼 정보 추가
                    symbol = os.path.basename(filepath).split('_')[0]
                    df['symbol'] = symbol
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 파일 읽기 실패 {filepath}: {e}")
            
            if all_data:
                # 데이터 통합
                merged_df = pd.concat(all_data, ignore_index=False)
                merged_df = merged_df.sort_index()
                
                # 통합 파일 저장
                merged_filename = f"merged_{timeframe}_{start_date}_to_{end_date}.csv"
                merged_filepath = os.path.join(self.data_dir, merged_filename)
                merged_df.to_csv(merged_filepath)
                
                print(f"✅ 통합 파일 저장: {merged_filepath}")
                print(f"📊 총 {len(merged_df)}개 캔들, {merged_df['symbol'].nunique()}개 심볼")
            
        except Exception as e:
            print(f"❌ 파일 통합 실패: {e}")
    
    def select_symbol_download(self, start_date: str, end_date: str, timeframe: str = '1h'):
        """심볼 선택 다운로드"""
        try:
            symbols = self.get_binance_usdtp_symbols()
            
            print(f"\n📋 사용 가능한 심볼 목록:")
            for i, symbol in enumerate(symbols, 1):
                print(f"{i:3d}. {symbol}")
            
            print(f"\n{len(symbols)+1:3d}. 전체 심볼 다운로드")
            print(f"{len(symbols)+2:3d}. 이전 메뉴로")
            
            while True:
                try:
                    choice = input(f"\n선택하세요 (1-{len(symbols)+2}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(symbols):
                        # 개별 심볼 선택
                        selected_symbol = symbols[choice_num - 1]
                        print(f"\n선택된 심볼: {selected_symbol}")
                        
                        filepath = self.download_symbol_data(selected_symbol, start_date, end_date, timeframe)
                        if filepath:
                            print(f"✅ 다운로드 완료: {filepath}")
                        break
                    elif choice_num == len(symbols) + 1:
                        # 전체 심볼 다운로드
                        print(f"\n전체 {len(symbols)}개 심볼 다운로드 시작...")
                        self.download_all_symbols(start_date, end_date, timeframe)
                        break
                    elif choice_num == len(symbols) + 2:
                        # 이전 메뉴로
                        break
                    else:
                        print("❌ 올바른 번호를 입력하세요.")
                        
                except ValueError:
                    print("❌ 숫자를 입력하세요.")
                    
        except Exception as e:
            print(f"❌ 심볼 선택 다운로드 실패: {e}")
    
    def get_available_data_files(self, timeframe: str = '1h'):
        """사용 가능한 데이터 파일 목록 반환"""
        try:
            files = []
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv') and timeframe in file:
                    filepath = os.path.join(self.data_dir, file)
                    # 파일 정보 가져오기
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    file_info = {
                        'filename': file,
                        'filepath': filepath,
                        'symbols': df['symbol'].nunique() if 'symbol' in df.columns else 1,
                        'candles': len(df),
                        'start_date': df.index.min().strftime('%Y-%m-%d'),
                        'end_date': df.index.max().strftime('%Y-%m-%d')
                    }
                    files.append(file_info)
            
            return sorted(files, key=lambda x: x['end_date'], reverse=True)
            
        except Exception as e:
            print(f"❌ 파일 목록 가져오기 실패: {e}")
            return []
    
    def show_data_files(self, timeframe: str = '1h'):
        """데이터 파일 목록 표시"""
        files = self.get_available_data_files(timeframe)
        
        if not files:
            print(f"❌ {timeframe} 시간프레임의 데이터 파일이 없습니다.")
            return None
        
        print(f"\n📁 사용 가능한 {timeframe} 데이터 파일:")
        print("="*60)
        for i, file_info in enumerate(files, 1):
            print(f"{i:2d}. {file_info['filename']}")
            print(f"    심볼: {file_info['symbols']}개, 캔들: {file_info['candles']:,}개")
            print(f"    기간: {file_info['start_date']} ~ {file_info['end_date']}")
            print()
        
        return files
    
    def extract_symbol_from_filename(self, filename: str) -> str:
        """파일명에서 심볼 추출"""
        try:
            # 파일명 형식: XLM_USDT_1h_20220101_20220630.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                symbol = f"{parts[0]}/{parts[1]}"
                return symbol
            return "Unknown"
        except:
            return "Unknown"
    
    def run_backtest_with_file(self, file_info: dict, strategy_name: str):
        """파일을 사용한 백테스트 실행"""
        try:
            print(f"\n🚀 백테스트 실행 중...")
            print(f"파일: {file_info['filename']}")
            print(f"전략: {strategy_name}")
            
            # 파일명에서 심볼 추출
            symbol = self.extract_symbol_from_filename(file_info['filename'])
            print(f"심볼: {symbol}")
            
            # 데이터 로드
            print(f"📁 파일 경로: {file_info['filepath']}")
            df = pd.read_csv(file_info['filepath'], index_col=0, parse_dates=True)
            print(f"📊 데이터 로드 완료: {len(df)}행, 컬럼: {list(df.columns)}")
            
            # 고급 통합전략의 경우 추가 처리
            if strategy_name == '고급 통합전략' and self.strategies.get('고급 통합전략'):
                print("🔧 고급 통합전략 초기화 중...")
                strategy = self.strategies['고급 통합전략']
                
                # 충분한 데이터가 있는 경우 ML 모델 훈련
                if len(df) >= 200:
                    print("🤖 머신러닝 모델 훈련 중...")
                    try:
                        if hasattr(strategy, 'ml_predictor') and strategy.ml_predictor:
                            strategy.ml_predictor.train_models(df)
                    except Exception as e:
                        print(f"⚠️ ML 모델 훈련 실패: {e}")
            
            # total_results 초기화
            total_results = []
            
            if 'symbol' in df.columns:
                # 통합 파일인 경우
                symbols = df['symbol'].unique()
                print(f"📊 {len(symbols)}개 심볼 백테스트 실행")
                
                for symbol in symbols:
                    symbol_data = df[df['symbol'] == symbol].copy()
                    symbol_data = symbol_data.drop('symbol', axis=1)
                    
                    if len(symbol_data) > 50:  # 최소 데이터 확인
                        print(f"  - {symbol} 처리 중...")
                        results = self.run_single_backtest(symbol_data, strategy_name, symbol)
                        if results:
                            total_results.append(results)
            
            else:
                # 단일 심볼 파일인 경우
                print(f"📊 단일 심볼 백테스트 실행: {symbol}")
                results = self.run_single_backtest(df, strategy_name, symbol)
                if results:
                    total_results.append(results)
                else:
                    print("❌ 백테스트 결과가 None입니다.")
            
            # 전체 결과 요약
            if total_results:
                self.show_summary_results(total_results)
            else:
                print("❌ 백테스트 결과가 없습니다.")
            
        except Exception as e:
            print(f"❌ 백테스트 실행 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def run_single_backtest(self, data: pd.DataFrame, strategy_name: str, symbol: str = "Unknown"):
        """단일 백테스트 실행"""
        try:
            print(f"🔧 백테스트 엔진 초기화 중...")
            # 백테스트 엔진 초기화
            engine = SimpleBacktestEngine(initial_balance=10000.0, commission=0.001)
            
            print(f"📈 기술적 지표 계산 중...")
            # 기술적 지표 계산
            data = engine.calculate_technical_indicators(data)
            print(f"📊 기술적 지표 계산 완료: {len(data)}행")
            
            print(f"🚀 백테스트 실행 중...")
            # 고급 통합전략인 경우 전략 객체 전달
            if strategy_name == "고급 통합전략" and self.strategies.get('고급 통합전략'):
                # 백테스트 엔진에 전략 객체 주입
                strategy_obj = self.strategies['고급 통합전략']
                results = engine.run_backtest_with_strategy(data, symbol, strategy_name, strategy_obj)
            else:
                # 백테스트 실행 (심볼명과 전략명 전달)
                results = engine.run_backtest(data, symbol, strategy_name)
            
            if results:
                print(f"✅ 백테스트 완료: {len(results.get('trades', []))}개 거래")
                results['symbol'] = symbol
                results['strategy'] = strategy_name
                return results
            else:
                print(f"❌ 백테스트 결과가 없습니다.")
                return None
            
        except Exception as e:
            print(f"❌ {symbol} 백테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def show_summary_results(self, results_list: list):
        """결과 요약 표시"""
        print("\n" + "="*60)
        print("📊 백테스트 결과 요약")
        print("="*60)
        
        total_trades = 0
        total_profit = 0
        total_symbols = len(results_list)
        
        for result in results_list:
            performance = result['performance']
            symbol = result.get('symbol', 'Unknown')
            
            print(f"\n{symbol}:")
            print(f"  수익률: {performance.get('total_return', 0):.2f}%")
            print(f"  거래 수: {performance.get('total_trades', 0)}")
            print(f"  승률: {performance.get('win_rate', 0):.1f}%")
            print(f"  최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
            
            total_trades += performance.get('total_trades', 0)
            total_profit += performance.get('total_return', 0)
        
        print(f"\n📈 전체 요약:")
        print(f"  평균 수익률: {total_profit/total_symbols:.2f}%")
        print(f"  총 거래 수: {total_trades}")
        print(f"  테스트 심볼: {total_symbols}개")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"results/summary_backtest_{timestamp}.json"
        
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 결과 저장: {results_filename}")
    
    def run(self):
        """메인 실행 루프"""
        while True:
            self.show_main_menu()
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == '1':
                self.handle_download_menu()
            elif choice == '2':
                self.handle_backtest_menu()
            elif choice == '3':
                print("\n👋 백테스트 시스템을 종료합니다.")
                break
            else:
                print("❌ 올바른 선택을 입력하세요.")
    
    def handle_download_menu(self):
        """다운로드 메뉴 처리"""
        while True:
            self.show_download_menu()
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == '1':
                # 심볼 목록 먼저 보여주고, 그 다음에 기간 설정
                symbols = self.get_binance_usdtp_symbols()
                if symbols:
                    start_date, end_date = self.get_date_range()
                    self.select_symbol_download(start_date, end_date)
                break
            elif choice == '2':
                start_date, end_date = self.get_date_range()
                self.download_all_symbols(start_date, end_date)
                break
            elif choice == '3':
                break
            else:
                print("❌ 올바른 선택을 입력하세요.")
    
    def handle_backtest_menu(self):
        """백테스트 메뉴 처리"""
        while True:
            self.show_strategy_menu()
            choice = input("선택하세요 (1-5): ").strip()
            
            if choice in ['1', '2', '3', '4']:
                strategy_map = {
                    '1': '급등 초입 진입전략',
                    '2': '고급 통합전략',
                    '3': '볼린저 밴드 브레이크아웃',
                    '4': '이동평균 교차'
                }
                strategy_name = strategy_map[choice]
                
                # 고급 통합전략인 경우 동적 로딩
                if choice == '2':
                    try:
                        from src.strategies.strategy_manager import StrategyManager
                        strategy_manager = StrategyManager()
                        self.strategies['고급 통합전략'] = strategy_manager.get_strategy()
                        print("✅ 고급 통합전략이 성공적으로 로드되었습니다.")
                        strategy_manager.print_strategy_summary()
                    except Exception as e:
                        print(f"❌ 고급 통합전략 로드 실패: {e}")
                        print("기본 급등 초입 진입전략을 사용합니다.")
                        strategy_name = '급등 초입 진입전략'
                
                # 데이터 파일 선택
                files = self.show_data_files('1h')
                if files:
                    try:
                        file_choice = input(f"파일을 선택하세요 (1-{len(files)}): ").strip()
                        file_idx = int(file_choice) - 1
                        
                        if 0 <= file_idx < len(files):
                            self.run_backtest_with_file(files[file_idx], strategy_name)
                            break
                        else:
                            print("❌ 올바른 파일 번호를 입력하세요.")
                    except ValueError:
                        print("❌ 숫자를 입력하세요.")
                break
            elif choice == '5':
                break
            else:
                print("❌ 올바른 선택을 입력하세요.")


def main():
    """메인 함수"""
    try:
        system = BacktestSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")


if __name__ == "__main__":
    main() 
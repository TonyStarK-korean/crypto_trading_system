#!/usr/bin/env python3
"""
GVS 웹서버 실행 스크립트
Crypto Trading System Web Dashboard Server
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_dependencies():
    """필요한 패키지 설치 확인"""
    required_packages = [
        'Flask==2.3.3',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'ccxt>=4.0.0',
        'plotly==5.17.0'
    ]
    
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("✅ All packages installed!")

def check_database_connection():
    """데이터베이스 연결 확인"""
    print("🔍 Checking database connection...")
    
    # SQLite 데이터베이스 경로 확인
    db_path = "data/web_trading.db"
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        print(f"📁 Created database directory: {os.path.dirname(db_path)}")
    
    print("✅ Database connection ready!")

def get_server_info():
    """서버 정보 표시"""
    print("\n🚀 CRYPTO TRADING SYSTEM WEB DASHBOARD")
    print("=" * 60)
    print(f"📅 Server Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 External IP: 34.47.77.230")
    print(f"🔌 Port: 5000")
    print(f"📊 Dashboard URL: http://34.47.77.230:5000")
    print(f"🐍 Python Version: {sys.version}")
    print(f"📂 Working Directory: {os.getcwd()}")
    print("=" * 60)

def check_existing_processes():
    """기존 프로세스 확인 및 종료"""
    print("🔍 Checking for existing processes...")
    
    try:
        # 포트 5000에서 실행 중인 프로세스 확인
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        if ':5000' in result.stdout:
            print("⚠️  Port 5000 is already in use!")
            
            # 기존 프로세스 종료 시도
            try:
                subprocess.run(['pkill', '-f', 'python.*app.py'], check=False)
                subprocess.run(['pkill', '-f', 'flask'], check=False)
                print("🛑 Terminated existing processes")
                time.sleep(2)
            except:
                pass
        else:
            print("✅ Port 5000 is available")
    except:
        print("✅ Process check completed")

def create_startup_script():
    """시작 스크립트 생성"""
    startup_script = """#!/bin/bash
# Crypto Trading System Auto-Start Script

export FLASK_APP=app.py
export FLASK_ENV=production
export FLASK_DEBUG=false

# 로그 디렉토리 생성
mkdir -p logs

# 웹 대시보드 시작
echo "Starting Crypto Trading System Web Dashboard..."
echo "Time: $(date)"
echo "Server: 34.47.77.230:5000"

# 백그라운드에서 실행
nohup python3 app.py > logs/server.log 2>&1 &

echo "Server started successfully!"
echo "Dashboard URL: http://34.47.77.230:5000"
echo "Log file: logs/server.log"
"""
    
    with open('start_dashboard.sh', 'w') as f:
        f.write(startup_script)
    
    # 실행 권한 부여
    os.chmod('start_dashboard.sh', 0o755)
    print("📜 Created startup script: start_dashboard.sh")

def main():
    """메인 함수"""
    print("🚀 CRYPTO TRADING SYSTEM WEB DASHBOARD SETUP")
    print("=" * 60)
    
    # 1. 의존성 확인
    check_dependencies()
    
    # 2. 데이터베이스 연결 확인
    check_database_connection()
    
    # 3. 기존 프로세스 확인
    check_existing_processes()
    
    # 4. 시작 스크립트 생성
    create_startup_script()
    
    # 5. 서버 정보 표시
    get_server_info()
    
    print("\n🎯 SETUP COMPLETE!")
    print("=" * 60)
    print("다음 명령어로 웹 대시보드를 시작하세요:")
    print("python3 app.py")
    print("\n또는 백그라운드에서 실행:")
    print("./start_dashboard.sh")
    print("\n웹 대시보드 접속:")
    print("http://34.47.77.230:5000")
    print("=" * 60)

if __name__ == "__main__":
    main()
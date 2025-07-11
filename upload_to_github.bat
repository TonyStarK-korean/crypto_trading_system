@echo off
echo 🚀 Uploading Crypto Trading System to GitHub...
echo Repository: https://github.com/TonyStarK-korean/crypto_trading_system

REM 현재 디렉토리를 프로젝트 디렉토리로 변경
cd /d "C:\projects\crypto_trading_system"

REM Git 초기화 (이미 초기화된 경우 무시)
git init

REM 원격 저장소 설정
git remote remove origin 2>nul
git remote add origin https://github.com/TonyStarK-korean/crypto_trading_system.git

REM 모든 파일 추가
git add .

REM 커밋 생성
git commit -m "Initial commit: Complete crypto trading system with web dashboard

Features:
- 🌐 Web dashboard with Flask
- 📊 Binance USDT pairs full scan backtest
- 🚀 Real-time trading system  
- 📈 Advanced backtesting with detailed trade records
- 🔧 Auto-deployment scripts for GVS server
- 🐳 Docker containerization
- 🤖 GitHub Actions CI/CD
- 📱 Responsive design with Bootstrap
- 🎯 Market phase analysis
- 💹 Leverage and position management
- 🔄 Auto-update system
- 🛡️ Security configurations
- 📋 Complete documentation"

REM GitHub에 푸시
echo 📤 Pushing to GitHub...
git branch -M main
git push -u origin main

echo ✅ Upload completed!
echo 🌐 Repository URL: https://github.com/TonyStarK-korean/crypto_trading_system
echo 📊 Web Dashboard will be available at: http://34.47.77.230:5000

pause
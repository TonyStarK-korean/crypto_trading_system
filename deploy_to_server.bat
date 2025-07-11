@echo off
echo 🚀 Deploying Crypto Trading System to GVS Server
echo Server: 34.47.77.230:5000

echo 📋 Deployment Steps:
echo 1. Upload code to GitHub
echo 2. Connect to GVS server via SSH
echo 3. Clone repository and run deployment script
echo 4. Configure 24/7 service

echo.
echo 🔗 SSH Connection Command:
echo ssh ubuntu@34.47.77.230

echo.
echo 📦 Server Setup Commands:
echo sudo apt update && sudo apt upgrade -y
echo sudo apt install -y python3 python3-pip git
echo git clone https://github.com/TonyStarK-korean/crypto_trading_system.git
echo cd crypto_trading_system/web_dashboard
echo chmod +x deploy.sh
echo ./deploy.sh

echo.
echo 🌐 After deployment, access via:
echo http://34.47.77.230:5000

echo.
echo 🔧 Service Management:
echo sudo systemctl status crypto-trading
echo sudo systemctl start crypto-trading
echo sudo systemctl stop crypto-trading
echo sudo systemctl restart crypto-trading
echo journalctl -u crypto-trading -f

echo.
echo 📊 Auto-Update:
echo python auto_update.py --daemon

pause
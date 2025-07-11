#!/usr/bin/env python3
"""
SSH 서버 설정 및 데이터베이스 연결 도구
GVS 웹서버 SSH 접속 및 설정 가이드
"""

import os
import sys
import subprocess
import getpass
import json
from datetime import datetime

def get_ssh_info():
    """SSH 접속 정보 확인"""
    print("🔍 SSH 접속 정보 확인")
    print("=" * 50)
    
    server_info = {
        'external_ip': '34.47.77.230',
        'port': 5000,
        'ssh_port': 22,
        'default_user': 'ubuntu',  # 일반적인 GCP/AWS 기본 사용자
        'service_user': 'crypto_trader'
    }
    
    print(f"🌐 External IP: {server_info['external_ip']}")
    print(f"🔌 Web Port: {server_info['port']}")
    print(f"🚪 SSH Port: {server_info['ssh_port']}")
    print(f"👤 Default User: {server_info['default_user']}")
    print(f"🤖 Service User: {server_info['service_user']}")
    
    return server_info

def check_ssh_connection():
    """SSH 연결 테스트"""
    print("\n🔗 SSH 연결 테스트")
    print("=" * 50)
    
    # SSH 키 확인
    ssh_key_path = os.path.expanduser('~/.ssh/id_rsa')
    if os.path.exists(ssh_key_path):
        print(f"✅ SSH 키 발견: {ssh_key_path}")
    else:
        print("❌ SSH 키를 찾을 수 없습니다.")
        print("SSH 키 생성 명령어:")
        print("ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'")
    
    # SSH 연결 명령어 제공
    server_info = get_ssh_info()
    print(f"\n📡 SSH 연결 명령어:")
    print(f"ssh ubuntu@{server_info['external_ip']}")
    print(f"ssh -i ~/.ssh/your_key.pem ubuntu@{server_info['external_ip']}")

def find_database_credentials():
    """데이터베이스 자격 증명 찾기"""
    print("\n🔍 데이터베이스 자격 증명 찾기")
    print("=" * 50)
    
    # 일반적인 데이터베이스 설정 파일 위치
    config_locations = [
        '/etc/postgresql/*/main/postgresql.conf',
        '/var/lib/postgresql/data/postgresql.conf',
        '/opt/postgres/postgresql.conf',
        '/home/ubuntu/.pgpass',
        '/root/.pgpass',
        '/etc/mysql/mysql.conf.d/mysqld.cnf',
        '/etc/mysql/my.cnf'
    ]
    
    env_variables = [
        'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'PGHOST', 'PGPORT',
        'DATABASE_URL', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_HOST'
    ]
    
    print("📁 데이터베이스 설정 파일 위치:")
    for location in config_locations:
        print(f"  • {location}")
    
    print("\n🔧 환경 변수 확인:")
    for env_var in env_variables:
        value = os.getenv(env_var)
        if value:
            print(f"  • {env_var}: {value}")
        else:
            print(f"  • {env_var}: 설정되지 않음")

def create_ssh_commands():
    """SSH 명령어 스크립트 생성"""
    print("\n📜 SSH 명령어 스크립트 생성")
    print("=" * 50)
    
    server_info = get_ssh_info()
    
    # SSH 연결 스크립트
    ssh_script = f"""#!/bin/bash
# Crypto Trading System SSH Connection Script

SERVER_IP="{server_info['external_ip']}"
SSH_PORT="{server_info['ssh_port']}"
USERNAME="{server_info['default_user']}"

echo "🚀 Connecting to Crypto Trading System Server..."
echo "Server: $SERVER_IP:$SSH_PORT"
echo "User: $USERNAME"
echo "Time: $(date)"

# SSH 연결 시도
echo "Attempting SSH connection..."
ssh -p $SSH_PORT $USERNAME@$SERVER_IP

# 연결 실패 시 대안
if [ $? -ne 0 ]; then
    echo "❌ SSH connection failed. Trying alternative methods..."
    echo "1. Check if SSH key is configured:"
    echo "   ssh-copy-id -i ~/.ssh/id_rsa.pub $USERNAME@$SERVER_IP"
    echo "2. Try with specific key:"
    echo "   ssh -i ~/.ssh/your_key.pem $USERNAME@$SERVER_IP"
    echo "3. Check firewall settings"
fi
"""
    
    with open('connect_ssh.sh', 'w') as f:
        f.write(ssh_script)
    
    os.chmod('connect_ssh.sh', 0o755)
    print("✅ SSH 연결 스크립트 생성: connect_ssh.sh")

def create_server_setup_script():
    """서버 설정 스크립트 생성"""
    print("\n⚙️  서버 설정 스크립트 생성")
    print("=" * 50)
    
    setup_script = """#!/bin/bash
# Crypto Trading System Server Setup Script

echo "🚀 Setting up Crypto Trading System on GVS Server..."
echo "Time: $(date)"
echo "Server: 34.47.77.230:5000"

# 시스템 업데이트
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Python 및 필수 패키지 설치
echo "🐍 Installing Python and dependencies..."
sudo apt install -y python3 python3-pip python3-venv git nginx

# PostgreSQL 설치 (선택사항)
echo "🗄️  Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

# 방화벽 설정
echo "🔒 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 사용자 생성
echo "👤 Creating service user..."
sudo adduser --disabled-password --gecos "" crypto_trader
sudo usermod -aG sudo crypto_trader

# 디렉토리 생성
echo "📁 Creating project directories..."
sudo mkdir -p /opt/crypto_trading_system
sudo chown crypto_trader:crypto_trader /opt/crypto_trading_system
cd /opt/crypto_trading_system

# Git 저장소 클론 (필요시)
# git clone https://github.com/your-repo/crypto_trading_system.git .

# Python 가상환경 생성
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install Flask pandas numpy ccxt plotly gunicorn

# 서비스 파일 생성
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/crypto-trading.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading System Web Dashboard
After=network.target

[Service]
Type=simple
User=crypto_trader
WorkingDirectory=/opt/crypto_trading_system
Environment=PATH=/opt/crypto_trading_system/venv/bin
ExecStart=/opt/crypto_trading_system/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading.service

# Nginx 설정 (선택사항)
echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/crypto-trading > /dev/null <<EOF
server {
    listen 80;
    server_name 34.47.77.230;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/crypto-trading /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

echo "✅ Server setup completed!"
echo "🌐 Dashboard URL: http://34.47.77.230:5000"
echo "📊 Direct access: http://34.47.77.230"
echo "🔧 Service management:"
echo "  • Start: sudo systemctl start crypto-trading"
echo "  • Stop: sudo systemctl stop crypto-trading"
echo "  • Status: sudo systemctl status crypto-trading"
echo "  • Logs: journalctl -u crypto-trading -f"
"""
    
    with open('server_setup.sh', 'w') as f:
        f.write(setup_script)
    
    os.chmod('server_setup.sh', 0o755)
    print("✅ 서버 설정 스크립트 생성: server_setup.sh")

def generate_database_config():
    """데이터베이스 설정 파일 생성"""
    print("\n🗄️  데이터베이스 설정 파일 생성")
    print("=" * 50)
    
    # PostgreSQL 설정
    pg_config = {
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "crypto_trading",
            "user": "crypto_trader",
            "password": "secure_password_here",
            "connection_string": "postgresql://crypto_trader:secure_password_here@localhost:5432/crypto_trading"
        },
        "sqlite": {
            "path": "/opt/crypto_trading_system/data/trading.db",
            "connection_string": "sqlite:///data/trading.db"
        }
    }
    
    with open('database_config.json', 'w') as f:
        json.dump(pg_config, f, indent=2)
    
    print("✅ 데이터베이스 설정 파일 생성: database_config.json")
    
    # PostgreSQL 설정 스크립트
    pg_setup = """#!/bin/bash
# PostgreSQL Setup for Crypto Trading System

echo "🗄️  Setting up PostgreSQL database..."

# PostgreSQL 서비스 시작
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 데이터베이스 및 사용자 생성
sudo -u postgres psql <<EOF
CREATE USER crypto_trader WITH PASSWORD 'secure_password_here';
CREATE DATABASE crypto_trading OWNER crypto_trader;
GRANT ALL PRIVILEGES ON DATABASE crypto_trading TO crypto_trader;
\\q
EOF

# 연결 테스트
echo "🔍 Testing database connection..."
psql -h localhost -U crypto_trader -d crypto_trading -c "SELECT version();"

echo "✅ PostgreSQL setup completed!"
echo "📊 Connection string: postgresql://crypto_trader:secure_password_here@localhost:5432/crypto_trading"
"""
    
    with open('setup_postgresql.sh', 'w') as f:
        f.write(pg_setup)
    
    os.chmod('setup_postgresql.sh', 0o755)
    print("✅ PostgreSQL 설정 스크립트 생성: setup_postgresql.sh")

def create_deployment_guide():
    """배포 가이드 생성"""
    print("\n📖 배포 가이드 생성")
    print("=" * 50)
    
    guide = """# Crypto Trading System 배포 가이드

## 🚀 GVS 웹서버 배포

### 1. SSH 접속
```bash
# 기본 접속
ssh ubuntu@34.47.77.230

# 키 파일 사용 시
ssh -i ~/.ssh/your_key.pem ubuntu@34.47.77.230
```

### 2. 서버 설정
```bash
# 설정 스크립트 실행
chmod +x server_setup.sh
./server_setup.sh
```

### 3. 데이터베이스 설정
```bash
# PostgreSQL 설정
chmod +x setup_postgresql.sh
./setup_postgresql.sh

# 또는 SQLite 사용 (간단한 옵션)
mkdir -p /opt/crypto_trading_system/data
```

### 4. 애플리케이션 배포
```bash
# 프로젝트 디렉토리로 이동
cd /opt/crypto_trading_system

# 가상환경 활성화
source venv/bin/activate

# 애플리케이션 파일 복사
# (GitHub, SCP, 또는 직접 업로드)

# 서비스 시작
sudo systemctl start crypto-trading
sudo systemctl enable crypto-trading
```

### 5. 웹 대시보드 접속
- URL: http://34.47.77.230:5000
- Direct: http://34.47.77.230 (Nginx 사용 시)

### 6. 시스템 관리
```bash
# 서비스 상태 확인
sudo systemctl status crypto-trading

# 로그 확인
journalctl -u crypto-trading -f

# 서비스 재시작
sudo systemctl restart crypto-trading
```

## 🗄️ 데이터베이스 연결

### PostgreSQL 연결
```python
DATABASE_URL = "postgresql://crypto_trader:secure_password_here@localhost:5432/crypto_trading"
```

### SQLite 연결 (간단한 옵션)
```python
DATABASE_URL = "sqlite:///data/trading.db"
```

## 🔧 문제 해결

### 포트 충돌 해결
```bash
# 포트 5000 사용 프로세스 확인
sudo lsof -i :5000

# 프로세스 종료
sudo kill -9 PID_NUMBER
```

### 방화벽 설정
```bash
# 필요한 포트 열기
sudo ufw allow 5000/tcp
sudo ufw reload
```

### 로그 확인
```bash
# 애플리케이션 로그
tail -f /opt/crypto_trading_system/logs/app.log

# 시스템 로그
journalctl -u crypto-trading -f
```

## 📊 모니터링

### 시스템 상태
```bash
# CPU, 메모리 사용량
htop

# 디스크 사용량
df -h

# 네트워크 연결
netstat -an | grep :5000
```

### 데이터베이스 상태
```bash
# PostgreSQL 상태
sudo systemctl status postgresql

# 데이터베이스 연결 테스트
psql -h localhost -U crypto_trader -d crypto_trading -c "SELECT NOW();"
```

## 🚀 자동 시작 설정

시스템 부팅 시 자동으로 서비스가 시작되도록 설정:

```bash
sudo systemctl enable crypto-trading
sudo systemctl enable postgresql
sudo systemctl enable nginx
```

## 🔒 보안 설정

### SSH 보안 강화
```bash
# SSH 키 인증만 허용
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes

sudo systemctl restart sshd
```

### 방화벽 강화
```bash
# 기본 정책 설정
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 필요한 포트만 허용
sudo ufw allow 22/tcp
sudo ufw allow 5000/tcp
sudo ufw enable
```

## 📱 원격 관리

### 영구 실행 (Screen 사용)
```bash
# Screen 설치
sudo apt install screen

# 새 screen 세션 시작
screen -S crypto_trading

# 애플리케이션 실행
python app.py

# Ctrl+A, D로 세션 분리
# screen -r crypto_trading로 재연결
```

### 원격 모니터링
```bash
# 실시간 로그 모니터링
tail -f /opt/crypto_trading_system/logs/app.log

# 시스템 리소스 모니터링
watch -n 1 'free -h && df -h'
```
"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("✅ 배포 가이드 생성: DEPLOYMENT_GUIDE.md")

def main():
    """메인 함수"""
    print("🔧 SSH 서버 설정 및 데이터베이스 연결 도구")
    print("=" * 60)
    
    # 1. SSH 정보 확인
    get_ssh_info()
    
    # 2. SSH 연결 확인
    check_ssh_connection()
    
    # 3. 데이터베이스 자격 증명 찾기
    find_database_credentials()
    
    # 4. SSH 명령어 스크립트 생성
    create_ssh_commands()
    
    # 5. 서버 설정 스크립트 생성
    create_server_setup_script()
    
    # 6. 데이터베이스 설정 생성
    generate_database_config()
    
    # 7. 배포 가이드 생성
    create_deployment_guide()
    
    print("\n✅ 모든 설정 파일이 생성되었습니다!")
    print("=" * 60)
    print("📁 생성된 파일들:")
    print("  • connect_ssh.sh - SSH 연결 스크립트")
    print("  • server_setup.sh - 서버 설정 스크립트")
    print("  • setup_postgresql.sh - PostgreSQL 설정 스크립트")
    print("  • database_config.json - 데이터베이스 설정")
    print("  • DEPLOYMENT_GUIDE.md - 상세 배포 가이드")
    print("\n🚀 다음 단계:")
    print("1. SSH로 서버에 접속: ./connect_ssh.sh")
    print("2. 서버 설정 실행: ./server_setup.sh")
    print("3. 데이터베이스 설정: ./setup_postgresql.sh")
    print("4. 웹 대시보드 접속: http://34.47.77.230:5000")
    print("=" * 60)

if __name__ == "__main__":
    main()
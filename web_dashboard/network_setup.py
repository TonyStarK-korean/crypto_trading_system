#!/usr/bin/env python3
"""
네트워크 설정 및 외부 접속 문제 해결 스크립트
Network Setup and External Access Configuration
"""

import os
import subprocess
import socket
import sys
import time

def check_port_availability(port):
    """포트 사용 가능 여부 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            print(f"✅ Port {port} is available")
            return True
    except OSError:
        print(f"❌ Port {port} is already in use")
        return False

def find_process_on_port(port):
    """포트를 사용하는 프로세스 찾기"""
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    pid = line.split()[-1]
                    print(f"🔍 Process using port {port}: PID {pid}")
                    return pid
        else:  # Linux/Unix
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.stdout:
                print(f"🔍 Process using port {port}:")
                print(result.stdout)
                return result.stdout.split('\n')[1].split()[1] if len(result.stdout.split('\n')) > 1 else None
    except Exception as e:
        print(f"Error checking port: {e}")
    return None

def configure_windows_firewall(port):
    """Windows 방화벽 설정"""
    try:
        # 인바운드 규칙 추가
        subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            f'name=CryptoTrading-{port}',
            'dir=in',
            'action=allow',
            'protocol=TCP',
            f'localport={port}'
        ], check=True)
        
        # 아웃바운드 규칙 추가
        subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            f'name=CryptoTrading-{port}-out',
            'dir=out',
            'action=allow',
            'protocol=TCP',
            f'localport={port}'
        ], check=True)
        
        print(f"✅ Windows Firewall configured for port {port}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to configure Windows Firewall for port {port}")
        return False

def configure_linux_firewall(port):
    """Linux 방화벽 설정 (UFW)"""
    try:
        # UFW 활성화
        subprocess.run(['sudo', 'ufw', '--force', 'enable'], check=True)
        
        # 포트 허용
        subprocess.run(['sudo', 'ufw', 'allow', str(port)], check=True)
        
        # 상태 확인
        result = subprocess.run(['sudo', 'ufw', 'status'], capture_output=True, text=True)
        print(f"✅ Linux Firewall (UFW) configured for port {port}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to configure Linux Firewall for port {port}")
        return False

def get_local_ip():
    """로컬 IP 주소 가져오기"""
    try:
        # 구글 DNS에 연결하여 로컬 IP 확인
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"

def check_internet_connection():
    """인터넷 연결 상태 확인"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print("✅ Internet connection is available")
        return True
    except OSError:
        print("❌ No internet connection")
        return False

def test_port_connectivity(port):
    """포트 연결성 테스트"""
    local_ip = get_local_ip()
    
    print(f"\n🌐 Testing port {port} connectivity...")
    print(f"Local IP: {local_ip}")
    
    # 로컬 연결 테스트
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex(('127.0.0.1', port))
            if result == 0:
                print(f"✅ Port {port} is accessible locally")
            else:
                print(f"❌ Port {port} is not accessible locally")
    except Exception as e:
        print(f"❌ Error testing local connection: {e}")

def create_test_server(port):
    """테스트 서버 실행"""
    try:
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<h1>Crypto Trading System Test Server</h1><p>Port connectivity test successful!</p>')
            
            def log_message(self, format, *args):
                pass  # 로그 출력 억제
        
        server = HTTPServer(('0.0.0.0', port), TestHandler)
        
        def run_server():
            server.serve_forever()
        
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        
        print(f"✅ Test server started on port {port}")
        return server
    except Exception as e:
        print(f"❌ Failed to start test server: {e}")
        return None

def main():
    """메인 함수"""
    print("🔧 Crypto Trading System Network Setup")
    print("=" * 50)
    
    PORT = 5000
    
    # 1. 인터넷 연결 확인
    print("\n1. Checking internet connection...")
    check_internet_connection()
    
    # 2. 포트 사용 가능 여부 확인
    print(f"\n2. Checking port {PORT} availability...")
    if not check_port_availability(PORT):
        pid = find_process_on_port(PORT)
        if pid:
            print(f"💡 To kill process: kill {pid} (Linux) or taskkill /PID {pid} /F (Windows)")
    
    # 3. 방화벽 설정
    print(f"\n3. Configuring firewall for port {PORT}...")
    if os.name == 'nt':  # Windows
        if not configure_windows_firewall(PORT):
            print("⚠️  Run as Administrator to configure Windows Firewall")
    else:  # Linux/Unix
        configure_linux_firewall(PORT)
    
    # 4. 로컬 IP 확인
    print(f"\n4. Network information:")
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    print(f"Localhost: http://127.0.0.1:{PORT}")
    print(f"LAN Access: http://{local_ip}:{PORT}")
    
    # 5. 테스트 서버 실행
    print(f"\n5. Starting test server...")
    test_server = create_test_server(PORT + 1)  # 다른 포트 사용
    
    if test_server:
        print(f"🌐 Test server URLs:")
        print(f"  • http://127.0.0.1:{PORT + 1}")
        print(f"  • http://{local_ip}:{PORT + 1}")
        
        print(f"\n⏱️  Test server will run for 30 seconds...")
        time.sleep(30)
        test_server.shutdown()
        print("✅ Test server stopped")
    
    # 6. 연결성 테스트
    print(f"\n6. Testing connectivity...")
    test_port_connectivity(PORT)
    
    print("\n" + "=" * 50)
    print("🎯 Network Setup Complete!")
    print("\n📋 Troubleshooting Steps:")
    print("1. Check if Flask app is running on 0.0.0.0:5000")
    print("2. Verify firewall rules allow port 5000")
    print("3. Check router port forwarding (for external access)")
    print("4. Ensure no other service is using port 5000")
    print("\n🔍 For external access:")
    print("1. Configure router port forwarding")
    print("2. Use public IP or domain name")
    print("3. Consider using ngrok for testing")

if __name__ == "__main__":
    main()
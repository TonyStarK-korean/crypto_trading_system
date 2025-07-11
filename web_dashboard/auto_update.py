#!/usr/bin/env python3
"""
자동 업데이트 시스템
Automatic Update System for Crypto Trading Dashboard
"""

import os
import subprocess
import sys
import time
import json
import requests
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoUpdater:
    def __init__(self, repo_url, branch='main', check_interval=300):
        self.repo_url = repo_url
        self.branch = branch
        self.check_interval = check_interval  # 5분마다 체크
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.service_name = 'crypto-trading'
        
    def check_for_updates(self):
        """GitHub에서 업데이트 확인"""
        try:
            # 현재 커밋 해시 가져오기
            current_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_dir
            ).decode().strip()
            
            # 원격 저장소에서 최신 커밋 해시 가져오기
            subprocess.run(['git', 'fetch', 'origin'], cwd=self.project_dir, check=True)
            latest_commit = subprocess.check_output(
                ['git', 'rev-parse', f'origin/{self.branch}'],
                cwd=self.project_dir
            ).decode().strip()
            
            if current_commit != latest_commit:
                logger.info(f"New update available: {latest_commit}")
                return True
            else:
                logger.info("No updates available")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking for updates: {e}")
            return False
    
    def backup_current_version(self):
        """현재 버전 백업"""
        try:
            backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            subprocess.run(['cp', '-r', self.project_dir, backup_dir], check=True)
            logger.info(f"Backup created: {backup_dir}")
            return backup_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def update_code(self):
        """코드 업데이트"""
        try:
            # Git pull
            subprocess.run(['git', 'pull', 'origin', self.branch], 
                          cwd=self.project_dir, check=True)
            logger.info("Code updated successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating code: {e}")
            return False
    
    def install_dependencies(self):
        """의존성 설치"""
        try:
            subprocess.run([
                'pip', 'install', '-r', 'requirements.txt'
            ], cwd=self.project_dir, check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def restart_service(self):
        """서비스 재시작"""
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', self.service_name], check=True)
            logger.info(f"Service {self.service_name} restarted successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error restarting service: {e}")
            return False
    
    def test_service_health(self):
        """서비스 상태 확인"""
        try:
            # HTTP 요청으로 서비스 상태 확인
            response = requests.get('http://localhost:5000/', timeout=10)
            if response.status_code == 200:
                logger.info("Service is healthy")
                return True
            else:
                logger.warning(f"Service returned status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Service health check failed: {e}")
            return False
    
    def send_notification(self, message):
        """알림 전송 (웹훅 또는 이메일)"""
        try:
            # 여기에 Discord, Slack, 이메일 알림 코드 추가
            logger.info(f"Notification: {message}")
            # 예: Discord 웹훅
            # webhook_url = "YOUR_DISCORD_WEBHOOK_URL"
            # requests.post(webhook_url, json={"content": message})
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def perform_update(self):
        """업데이트 수행"""
        logger.info("Starting update process...")
        
        # 1. 백업 생성
        backup_dir = self.backup_current_version()
        if not backup_dir:
            logger.error("Backup failed, aborting update")
            return False
        
        # 2. 코드 업데이트
        if not self.update_code():
            logger.error("Code update failed, aborting")
            return False
        
        # 3. 의존성 설치
        if not self.install_dependencies():
            logger.error("Dependencies installation failed, aborting")
            return False
        
        # 4. 서비스 재시작
        if not self.restart_service():
            logger.error("Service restart failed, aborting")
            return False
        
        # 5. 건강 상태 확인
        time.sleep(10)  # 서비스 시작 대기
        if not self.test_service_health():
            logger.error("Service health check failed after update")
            self.send_notification("🚨 Service health check failed after update!")
            return False
        
        logger.info("Update completed successfully")
        self.send_notification("✅ Crypto Trading System updated successfully!")
        return True
    
    def run(self):
        """자동 업데이트 실행"""
        logger.info("Starting auto-updater...")
        
        while True:
            try:
                if self.check_for_updates():
                    self.perform_update()
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Auto-updater stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # 에러 발생 시 1분 대기

def main():
    """메인 함수"""
    # 설정
    REPO_URL = "https://github.com/TonyStarK-korean/crypto_trading_system.git"
    BRANCH = "main"
    CHECK_INTERVAL = 300  # 5분
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    # 자동 업데이트 시작
    updater = AutoUpdater(REPO_URL, BRANCH, CHECK_INTERVAL)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        # 데몬 모드로 실행
        updater.run()
    else:
        # 한 번만 업데이트 확인
        if updater.check_for_updates():
            updater.perform_update()

if __name__ == "__main__":
    main()
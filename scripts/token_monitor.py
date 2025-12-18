"""
Token Refresh Monitor
Monitors Schwab API token expiration and sends alerts when refresh is needed
Run this as a separate cron job or systemd timer
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/token_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TokenMonitor:
    """Monitor Schwab API token expiration and send alerts"""
    
    def __init__(self):
        self.token_file = Path(__file__).parent.parent / 'schwab_client.json'
        
    def load_token_info(self):
        """Load token information from file"""
        try:
            with open(self.token_file, 'r') as f:
                config = json.load(f)
                return config.get('token', {})
        except Exception as e:
            logger.error(f"Failed to load token file: {e}")
            return None
    
    def check_token_expiration(self):
        """Check if tokens are expiring soon"""
        token = self.load_token_info()
        
        if not token:
            return {
                'status': 'ERROR',
                'message': 'Cannot load token file',
                'action_needed': True
            }
        
        current_time = datetime.now(timezone.utc)
        
        # Check access token
        access_expires_at = token.get('expires_at')
        if access_expires_at:
            access_expires = datetime.fromtimestamp(int(access_expires_at), timezone.utc)
            access_time_left = (access_expires - current_time).total_seconds()
            
            if access_time_left < 300:  # Less than 5 minutes
                logger.warning(f"Access token expires in {access_time_left/60:.1f} minutes")
        
        # Check refresh token (most critical)
        refresh_token_created_at = token.get('refresh_token_created_at')
        refresh_token_expires_in = token.get('refresh_token_expires_in', 604800)  # Default 7 days
        
        if not refresh_token_created_at:
            logger.warning("⚠️  Status: UNKNOWN - Cannot determine refresh token age")
            logger.warning("⚠️  The token file is missing 'refresh_token_created_at' timestamp")
            logger.warning("⚠️  This timestamp was added in the recent update")
            logger.warning("⚠️  Action: Run ./scripts/refresh_worker_auth.sh to regenerate tokens")
            logger.warning("⚠️  After refresh, monitoring will work properly")
            return {
                'status': 'UNKNOWN',
                'message': 'Token file missing refresh_token_created_at. Run refresh_worker_auth.sh to update.',
                'action_needed': True
            }
        
        if refresh_token_created_at:
            refresh_created = datetime.fromtimestamp(int(refresh_token_created_at), timezone.utc)
            refresh_expires = refresh_created.timestamp() + refresh_token_expires_in
            refresh_expires_dt = datetime.fromtimestamp(refresh_expires, timezone.utc)
            refresh_time_left = (refresh_expires_dt - current_time).total_seconds()
            
            days_left = refresh_time_left / 86400  # Convert to days
            hours_left = refresh_time_left / 3600  # Convert to hours
            
            logger.info(f"Refresh token expires in {days_left:.2f} days ({hours_left:.1f} hours)")
            
            if days_left < 1:  # Less than 1 day
                return {
                    'status': 'CRITICAL',
                    'message': f'Refresh token expires in {hours_left:.1f} hours! Manual re-authentication required ASAP.',
                    'action_needed': True,
                    'expires_in_hours': hours_left,
                    'expires_at': refresh_expires_dt.isoformat()
                }
            elif days_left < 2:  # Less than 2 days
                return {
                    'status': 'WARNING',
                    'message': f'Refresh token expires in {days_left:.2f} days. Please re-authenticate soon.',
                    'action_needed': True,
                    'expires_in_days': days_left,
                    'expires_at': refresh_expires_dt.isoformat()
                }
            else:
                return {
                    'status': 'OK',
                    'message': f'Refresh token is valid for {days_left:.2f} more days',
                    'action_needed': False,
                    'expires_in_days': days_left,
                    'expires_at': refresh_expires_dt.isoformat()
                }
        else:
            # No refresh token created timestamp, check if refresh_token exists
            if 'refresh_token' not in token:
                return {
                    'status': 'CRITICAL',
                    'message': 'No refresh token found! Re-authentication required.',
                    'action_needed': True
                }
            
            return {
                'status': 'UNKNOWN',
                'message': 'Cannot determine refresh token expiration (missing created_at)',
                'action_needed': False
            }
    
    def send_alert(self, status_info):
        """Send alert notification (can be extended to email, SMS, Slack, etc.)"""
        if not status_info['action_needed']:
            return
        
        message = f"""
🚨 SCHWAB API TOKEN ALERT 🚨

Status: {status_info['status']}
Message: {status_info['message']}

Action Required: Run the token refresh script
Command: ./scripts/refresh_worker_auth.sh

This will:
1. Stop the worker
2. Re-authenticate with Schwab API
3. Copy fresh tokens to the droplet
4. Restart the worker

Do this ASAP to avoid service interruption!
        """
        
        logger.warning(message)
        
        # You can add email, Slack, Discord notifications here
        # Example: self.send_email(message)
        # Example: self.send_discord_webhook(message)
        
        return message
    
    def send_email(self, message):
        """Send email alert (configure with your SMTP settings)"""
        # This is a placeholder - configure with actual SMTP settings
        pass

def main():
    """Main monitoring function"""
    monitor = TokenMonitor()
    
    logger.info("=" * 60)
    logger.info("Schwab API Token Monitor")
    logger.info("=" * 60)
    
    status = monitor.check_token_expiration()
    
    logger.info(f"Status: {status['status']}")
    logger.info(f"Message: {status['message']}")
    
    if status['action_needed']:
        monitor.send_alert(status)
    
    return status['status']

if __name__ == "__main__":
    status = main()
    # Exit with error code if action needed
    if status in ['CRITICAL', 'WARNING']:
        sys.exit(1)
    else:
        sys.exit(0)

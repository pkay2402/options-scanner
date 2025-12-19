"""
Health Monitor - Monitors system health and sends Discord alerts for issues
Run this as a separate cron job every 15 minutes
"""

import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
import discord
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitor system health and alert on issues"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent.parent / 'data' / 'market_cache.db'
        self.token_path = Path(__file__).parent.parent / 'schwab_client.json'
        self.discord_webhook = self._load_webhook()
        
    def _load_webhook(self):
        """Load Discord webhook from environment or config"""
        import os
        webhook_url = os.environ.get('DISCORD_HEALTH_WEBHOOK')
        if not webhook_url:
            config_path = Path(__file__).parent.parent / 'config' / 'health_monitor.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    webhook_url = config.get('discord_webhook')
        return webhook_url
    
    def check_token_expiration(self):
        """Check if Schwab token is expiring soon"""
        issues = []
        
        if not self.token_path.exists():
            issues.append({
                'severity': 'CRITICAL',
                'component': 'Schwab API Token',
                'message': 'Token file not found',
                'action': 'Manual reauthorization required'
            })
            return issues
        
        try:
            with open(self.token_path) as f:
                data = json.load(f)
                token = data.get('token', {})
                
                # Check access token expiration
                if 'expires_at' in token:
                    expires = datetime.fromtimestamp(token['expires_at'])
                    time_left = expires - datetime.now()
                    
                    if time_left.total_seconds() < 0:
                        issues.append({
                            'severity': 'CRITICAL',
                            'component': 'Schwab Access Token',
                            'message': f'Access token expired {-time_left.total_seconds()/3600:.1f} hours ago',
                            'action': 'Should auto-refresh, check worker logs'
                        })
                
                # Check refresh token expiration (7 days)
                if 'refresh_token_created_at' in token:
                    created = datetime.fromtimestamp(token['refresh_token_created_at'])
                    age = datetime.now() - created
                    days_left = 7 - age.days
                    
                    if days_left < 0:
                        issues.append({
                            'severity': 'CRITICAL',
                            'component': 'Schwab Refresh Token',
                            'message': f'Refresh token expired {-days_left} days ago',
                            'action': 'MANUAL REAUTHORIZATION REQUIRED IMMEDIATELY'
                        })
                    elif days_left <= 2:
                        issues.append({
                            'severity': 'WARNING',
                            'component': 'Schwab Refresh Token',
                            'message': f'Refresh token expires in {days_left} days',
                            'action': f'Prepare to reauthorize by {(created + timedelta(days=7)).strftime("%Y-%m-%d")}'
                        })
        
        except Exception as e:
            issues.append({
                'severity': 'ERROR',
                'component': 'Token Check',
                'message': f'Failed to check token: {e}',
                'action': 'Check token file format'
            })
        
        return issues
    
    def check_database_updates(self):
        """Check if database is being updated regularly"""
        issues = []
        
        if not self.db_path.exists():
            issues.append({
                'severity': 'CRITICAL',
                'component': 'Database',
                'message': 'Database file not found',
                'action': 'Check worker service status'
            })
            return issues
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check last watchlist update
            cursor.execute("""
                SELECT value, updated_at 
                FROM cache_metadata 
                WHERE key = 'watchlist'
            """)
            result = cursor.fetchone()
            
            if result:
                last_update = datetime.fromisoformat(result[1])
                age = datetime.now() - last_update
                
                if age.total_seconds() > 900:  # 15 minutes
                    issues.append({
                        'severity': 'CRITICAL',
                        'component': 'Watchlist Updates',
                        'message': f'Last update {age.total_seconds()/60:.0f} minutes ago',
                        'action': 'Check market-data-worker service'
                    })
            else:
                issues.append({
                    'severity': 'WARNING',
                    'component': 'Watchlist',
                    'message': 'No watchlist metadata found',
                    'action': 'Check if worker has run at least once'
                })
            
            # Check whale flows
            cursor.execute("""
                SELECT value, updated_at 
                FROM cache_metadata 
                WHERE key = 'whale_flows'
            """)
            result = cursor.fetchone()
            
            if result:
                last_update = datetime.fromisoformat(result[1])
                age = datetime.now() - last_update
                
                if age.total_seconds() > 900:  # 15 minutes
                    issues.append({
                        'severity': 'WARNING',
                        'component': 'Whale Flow Detection',
                        'message': f'Last scan {age.total_seconds()/60:.0f} minutes ago',
                        'action': 'Check if worker is scanning whale flows'
                    })
            
            # Check database size
            cursor.execute("SELECT COUNT(*) FROM watchlist")
            watchlist_count = cursor.fetchone()[0]
            
            if watchlist_count == 0:
                issues.append({
                    'severity': 'CRITICAL',
                    'component': 'Watchlist Data',
                    'message': 'Watchlist table is empty',
                    'action': 'Check worker logs for errors'
                })
            
            conn.close()
            
        except Exception as e:
            issues.append({
                'severity': 'ERROR',
                'component': 'Database Check',
                'message': f'Failed to check database: {e}',
                'action': 'Check database file integrity'
            })
        
        return issues
    
    def check_memory_usage(self):
        """Check if services are approaching memory limits"""
        issues = []
        
        try:
            import subprocess
            
            # Check worker memory
            result = subprocess.run(
                ['systemctl', 'show', 'market-data-worker', '--property=MemoryCurrent,MemoryHigh,MemoryMax'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                memory_info = {}
                for line in lines:
                    key, value = line.split('=')
                    if value != '[not set]':
                        memory_info[key] = int(value)
                
                if 'MemoryCurrent' in memory_info and 'MemoryHigh' in memory_info:
                    current = memory_info['MemoryCurrent'] / (1024 * 1024)  # MB
                    high = memory_info['MemoryHigh'] / (1024 * 1024)
                    
                    if current > high * 0.95:
                        issues.append({
                            'severity': 'WARNING',
                            'component': 'Worker Memory',
                            'message': f'Using {current:.0f}MB (95%+ of {high:.0f}MB limit)',
                            'action': 'Consider increasing memory limit'
                        })
            
        except Exception as e:
            logger.debug(f"Could not check memory usage: {e}")
        
        return issues
    
    async def send_discord_alert(self, issues):
        """Send health issues to Discord"""
        if not self.discord_webhook or not issues:
            return
        
        webhook = discord.Webhook.from_url(
            self.discord_webhook,
            adapter=discord.RequestsWebhookAdapter()
        )
        
        # Group by severity
        critical = [i for i in issues if i['severity'] == 'CRITICAL']
        warnings = [i for i in issues if i['severity'] == 'WARNING']
        errors = [i for i in issues if i['severity'] == 'ERROR']
        
        color = 0xFF0000 if critical else (0xFF9900 if warnings else 0x999999)
        
        embed = discord.Embed(
            title="🚨 System Health Alert" if critical else "⚠️ System Health Warning",
            description=f"Found {len(issues)} issue(s) requiring attention",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        for issue in critical + warnings + errors:
            emoji = "🔴" if issue['severity'] == 'CRITICAL' else "🟡" if issue['severity'] == 'WARNING' else "⚪"
            embed.add_field(
                name=f"{emoji} {issue['component']}",
                value=f"**Issue:** {issue['message']}\n**Action:** {issue['action']}",
                inline=False
            )
        
        embed.set_footer(text="Health Monitor")
        
        webhook.send(embed=embed)
        logger.info(f"Sent Discord alert for {len(issues)} issues")
    
    def run(self):
        """Run all health checks"""
        logger.info("Starting health check...")
        
        all_issues = []
        
        # Run all checks
        all_issues.extend(self.check_token_expiration())
        all_issues.extend(self.check_database_updates())
        all_issues.extend(self.check_memory_usage())
        
        if all_issues:
            logger.warning(f"Found {len(all_issues)} health issues:")
            for issue in all_issues:
                logger.warning(f"  {issue['severity']}: {issue['component']} - {issue['message']}")
            
            # Send to Discord
            if self.discord_webhook:
                asyncio.run(self.send_discord_alert(all_issues))
        else:
            logger.info("✓ All health checks passed")
        
        return len(all_issues)


if __name__ == '__main__':
    monitor = HealthMonitor()
    exit_code = monitor.run()
    sys.exit(0 if exit_code == 0 else 1)

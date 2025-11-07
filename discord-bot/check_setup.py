#!/usr/bin/env python3
"""
Pre-flight check script for Discord bot
Verifies all requirements are met before running the bot
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status=True):
    """Print colored status message"""
    icon = f"{GREEN}✅{RESET}" if status else f"{RED}❌{RESET}"
    print(f"{icon} {message}")

def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {message}{RESET}")

def print_info(message):
    """Print info message"""
    print(f"{BLUE}ℹ️  {message}{RESET}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_status(f"Python version: {version.major}.{version.minor}.{version.micro} (3.11+ recommended)", False)
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'discord': 'discord.py',
        'dotenv': 'python-dotenv',
        'pandas': 'pandas',
        'plotly': 'plotly',
        'kaleido': 'kaleido'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print_status(f"Package '{package}' installed")
        except ImportError:
            print_status(f"Package '{package}' NOT installed", False)
            missing.append(package)
    
    return len(missing) == 0, missing

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print_status(".env file exists", False)
        print_info("Copy .env.template to .env and fill in your credentials")
        return False
    
    print_status(".env file exists")
    
    # Read .env and check for required variables
    required_vars = [
        'DISCORD_BOT_TOKEN',
        'SCHWAB_CLIENT_ID',
        'SCHWAB_CLIENT_SECRET',
        'SCHWAB_REDIRECT_URI'
    ]
    
    env_content = env_path.read_text()
    missing_vars = []
    
    for var in required_vars:
        if f'{var}=' in env_content:
            # Check if it has a value (not just the template placeholder)
            for line in env_content.split('\n'):
                if line.startswith(f'{var}='):
                    value = line.split('=', 1)[1].strip()
                    if value and not value.startswith('your_'):
                        print_status(f"  {var} configured")
                    else:
                        print_status(f"  {var} NOT configured", False)
                        missing_vars.append(var)
                    break
        else:
            print_status(f"  {var} NOT found", False)
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def check_schwab_client():
    """Check if schwab_client.json exists"""
    client_path = Path('../schwab_client.json')
    
    if client_path.exists():
        print_status("schwab_client.json exists (OAuth tokens present)")
        return True
    else:
        print_warning("schwab_client.json NOT found")
        print_info("You'll need to complete OAuth flow on first run")
        return True  # Not a blocker, just a warning

def check_src_directory():
    """Check if src/ directory is accessible"""
    src_path = Path('../src/api/schwab_client.py')
    
    if src_path.exists():
        print_status("Parent src/ directory accessible")
        return True
    else:
        print_status("Parent src/ directory NOT accessible", False)
        return False

def main():
    """Run all pre-flight checks"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🚀 Discord Bot Pre-Flight Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    checks = []
    
    # Python version
    print(f"{BLUE}[1/5] Checking Python version...{RESET}")
    checks.append(check_python_version())
    print()
    
    # Dependencies
    print(f"{BLUE}[2/5] Checking dependencies...{RESET}")
    deps_ok, missing = check_dependencies()
    checks.append(deps_ok)
    if not deps_ok:
        print_info(f"Install missing packages: pip install {' '.join(missing)}")
    print()
    
    # Environment file
    print(f"{BLUE}[3/5] Checking .env configuration...{RESET}")
    env_ok, missing_vars = check_env_file()
    checks.append(env_ok)
    if not env_ok:
        print_info(f"Configure these variables in .env: {', '.join(missing_vars)}")
    print()
    
    # Schwab client
    print(f"{BLUE}[4/5] Checking Schwab authentication...{RESET}")
    checks.append(check_schwab_client())
    print()
    
    # Src directory
    print(f"{BLUE}[5/5] Checking parent src/ access...{RESET}")
    checks.append(check_src_directory())
    print()
    
    # Summary
    print(f"{BLUE}{'='*60}{RESET}")
    if all(checks):
        print(f"{GREEN}✅ All checks passed! Ready to run the bot.{RESET}")
        print(f"\n{GREEN}Run: ./start.sh{RESET}")
        print(f"{GREEN}Or:  python -m bot.main{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ Some checks failed. Please fix the issues above.{RESET}\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())

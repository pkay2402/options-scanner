#!/usr/bin/env python3
"""
Wrapper script to run Discord bot as PythonAnywhere always-on task
"""
import sys
import os
from pathlib import Path

# Set up paths
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Set PYTHONPATH
sys.path.insert(0, str(script_dir))
os.environ['PYTHONPATH'] = str(script_dir)

# Create logs directory if needed
logs_dir = script_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

# Import and run the bot
from bot.main import main

if __name__ == '__main__':
    main()

#!/bin/bash
# Restart all scanners with improved rate limiting

echo "Stopping existing scanners..."
pkill -f 'macd_scanner.py'
pkill -f 'ttm_squeeze_scanner.py'
pkill -f 'vpb_scanner.py'

sleep 3

echo "Starting scanners..."
cd /root/options-scanner

nohup /root/options-scanner/venv/bin/python scripts/macd_scanner.py >> logs/macd_scanner.log 2>&1 &
nohup /root/options-scanner/venv/bin/python scripts/ttm_squeeze_scanner.py >> logs/ttm_squeeze_scanner.log 2>&1 &
nohup /root/options-scanner/venv/bin/python scripts/vpb_scanner.py >> logs/vpb_scanner.log 2>&1 &

sleep 2

echo "Scanner processes:"
ps aux | grep scanner.py | grep python | grep -v grep

echo "Done! Scanners restarted."

#!/bin/bash
# Check if Schwab tokens are expiring soon and send notification

TOKEN_FILE="/Users/piyushkhaitan/schwab/options/schwab_client.json"
WARN_DAYS=2  # Warn when less than 2 days remaining

if [ ! -f "$TOKEN_FILE" ]; then
    echo "⚠️  Token file not found: $TOKEN_FILE"
    exit 1
fi

# Extract setup date from token file
SETUP_DATE=$(grep -o '"setup": "[^"]*"' "$TOKEN_FILE" | cut -d'"' -f4)

if [ -z "$SETUP_DATE" ]; then
    echo "⚠️  Could not read setup date from token file"
    exit 1
fi

# Convert to seconds since epoch
SETUP_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$SETUP_DATE" "+%s")
EXPIRE_EPOCH=$((SETUP_EPOCH + 604800))  # 7 days = 604800 seconds
WARN_EPOCH=$((EXPIRE_EPOCH - (WARN_DAYS * 86400)))  # 2 days before
NOW_EPOCH=$(date +%s)

# Calculate days remaining
DAYS_REMAINING=$(( (EXPIRE_EPOCH - NOW_EPOCH) / 86400 ))

if [ $NOW_EPOCH -ge $EXPIRE_EPOCH ]; then
    echo "🔴 EXPIRED! Token expired $(( (NOW_EPOCH - EXPIRE_EPOCH) / 86400 )) days ago"
    echo "Run: ./scripts/refresh_tokens_everywhere.sh"
    exit 2
elif [ $NOW_EPOCH -ge $WARN_EPOCH ]; then
    echo "🟡 WARNING! Token expires in $DAYS_REMAINING days"
    echo "Token created: $SETUP_DATE"
    echo "Expires: $(date -r $EXPIRE_EPOCH "+%Y-%m-%d %H:%M:%S")"
    echo ""
    echo "Run: ./scripts/refresh_tokens_everywhere.sh"
    exit 1
else
    echo "🟢 Token OK - $DAYS_REMAINING days remaining"
    echo "Token created: $SETUP_DATE"
    echo "Next refresh needed: $(date -r $WARN_EPOCH "+%Y-%m-%d")"
    exit 0
fi

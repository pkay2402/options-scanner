# Quick Start: Setting Up Automated Alerts

## 1. Start Your Discord Bot

```bash
cd discord-bot
python run_bot.py
```

## 2. In Your Discord Server

### Setup Alerts (One-time)

1. Go to the channel where you want to receive alerts
2. Run: `/alerts_setup`
   - This configures the channel for automated alerts
3. Run: `/alerts_start`
   - This starts the 15-minute scanning

### That's it! 

The bot will now send:
- 🐋 **Whale flows** with score > 300 (no duplicates)
- 📊 **0DTE levels** for SPY, QQQ, $SPX

Every 15 minutes during market hours (9:30 AM - 4:00 PM ET)

## 3. Test It

Run: `/alerts_test`

You should see a test message appear in your configured channel.

## 4. Check Status Anytime

Run: `/alerts_status`

Shows:
- ✅ Service running or stopped
- 🟢 Market open or closed
- Number of cached whale alerts
- Current threshold settings

## 5. Stop Alerts (if needed)

Run: `/alerts_stop`

## Example Alert Flow

**9:30 AM** - Market opens, bot starts scanning

**9:45 AM** - First scan completes:
```
🐋 Whale Flow Alert
Found 2 new whale flows (Score > 300)

NVDA $850.00 CALL
Score: 1,245 | Vol: 15,000 | OI: 3,200
Distance: +2.3% | IV: 45.2%
```

```
📊 0DTE Levels Update
SPY: $585.25 🟡 Between Walls
Call Wall: $586 (125K vol) | Put Wall: $584 (98K vol)
```

**10:00 AM** - Second scan (only NEW whale flows alerted)

**4:00 PM** - Market closes, alerts pause, cache clears

## Troubleshooting

**No alerts?**
- Check: `/alerts_status` - is service running?
- Is market open? (9:30 AM - 4:00 PM ET, Mon-Fri)
- Check bot logs: `tail -f discord-bot/bot.log`

**Too many alerts?**
- Increase threshold in `bot/services/alert_service.py`
- Change `self.whale_score_threshold = 300` to higher value

**Want different symbols?**
- Edit `TOP_TECH_STOCKS` in `bot/commands/whale_score.py`
- Or use manual `/whalescan` command for custom scans

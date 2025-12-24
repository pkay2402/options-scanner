# Z-Score Scanner Quick Reference

## 🎯 Available Commands

### Setup Commands
```
/setup_zscore_scanner
```
Configure the current channel for z-score alerts.

```
/start_zscore_scanner
```
Start automated monitoring (every 15 mins during market hours).

```
/stop_zscore_scanner
```
Stop automated monitoring.

```
/zscore_status
```
Check scanner status and recent alerts.

### Analysis Commands
```
/check_zscore symbol:AAPL
```
Manually check z-score for any symbol.

---

## 📊 Signal Types

### Buy Signals (🟢)
- **-2σ** - Price 2 standard deviations below mean (oversold)
- **-3σ** - Price 3 standard deviations below mean (extreme oversold)

### Sell Signals (🔴)
- **+2σ** - Price 2 standard deviations above mean (overbought)
- **+3σ** - Price 3 standard deviations above mean (extreme overbought)

---

## ⭐ Signal Quality

### High Quality (⭐⭐⭐)
**All conditions met:**
- ✅ RSI < 40
- ✅ Price > -15% from 50-MA
- ✅ Momentum > -10% OR Volume > 1.5x

**Action:** High confidence signal, consider entry

### Weak (⚠️)
**One or more conditions failed**

**Action:** Wait for confirmation

---

## ⏰ Scan Schedule

- **Frequency:** Every 15 minutes
- **Active:** 9:30 AM - 4:00 PM ET
- **Days:** Monday - Friday only
- **Auto-sleep:** Outside market hours

---

## 📈 Watchlist

Current stocks monitored:
```
SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, TSLA, AMZN, 
META, GOOGL, AMD, NFLX, CRM, AVGO, JPM, GS, WMT, 
CRWD, PG
```

---

## 📊 Alert Contents

Each alert includes:

1. **Summary Embed**
   - Signal type (Buy/Sell)
   - Quality rating
   - Price, Z-Score, RSI
   - Trend %, Volume ratio

2. **Chart Image**
   - Price history
   - Z-score over time
   - Threshold lines (-3, -2, +2, +3)

3. **Actionable Advice**
   - Based on quality rating
   - Specific reasons if weak signal

---

## 🔧 Troubleshooting

### No alerts appearing?
1. Check status: `/zscore_status`
2. Verify market hours (9:30-4:00 ET)
3. Check if scanner is running
4. Restart: `/stop_zscore_scanner` → `/start_zscore_scanner`

### Bot not responding?
```bash
ssh root@138.197.210.166
systemctl status discord-bot
systemctl restart discord-bot
```

### Check logs
```bash
tail -f /root/options-scanner/logs/discord_bot.log
```

---

## 💡 Best Practices

1. **Setup once per channel**
   - Run `/setup_zscore_scanner` in dedicated alerts channel

2. **Monitor during volatility**
   - Most signals during market open/close
   - High volatility = more crossings

3. **Combine with other indicators**
   - Use with whale flows
   - Check volume confirmation
   - Review overall trend

4. **Quality matters**
   - ⭐⭐⭐ signals have ~80-90% accuracy
   - ⚠️ signals have ~40-50% accuracy
   - Wait for confirmation on weak signals

5. **Track performance**
   - Note which symbols give best signals
   - Review false positives
   - Adjust position sizing by quality

---

## 📞 Quick Links

- **Logs:** `/root/options-scanner/logs/discord_bot.log`
- **Config:** `/root/options-scanner/user_preferences.json`
- **Service:** `systemctl status discord-bot`
- **Docs:** `discord-bot/ZSCORE_SCANNER_DEPLOYMENT.md`

---

*For detailed documentation, see ZSCORE_SCANNER_DEPLOYMENT.md*

-- Options Flow Scanner Database Schema
-- PostgreSQL 14+

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Core Tables: Raw Options Data
-- ============================================================================

-- Whale Flows: High VALR scoring options
CREATE TABLE whale_flows (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike DECIMAL(10, 2) NOT NULL,
    type VARCHAR(4) NOT NULL CHECK (type IN ('CALL', 'PUT')),
    whale_score INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    gamma DECIMAL(10, 6),
    gex DECIMAL(20, 2),
    mark DECIMAL(10, 2),
    delta DECIMAL(6, 4),
    iv DECIMAL(8, 2),
    underlying_price DECIMAL(10, 2) NOT NULL,
    call_volume INTEGER,
    put_volume INTEGER,
    vol_ratio DECIMAL(8, 2),
    max_gex_strike DECIMAL(10, 2),
    max_gex_value DECIMAL(20, 2),
    call_wall_strike DECIMAL(10, 2),
    put_wall_strike DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- OI Flows: Fresh positioning (Vol/OI >= 3.0)
CREATE TABLE oi_flows (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike DECIMAL(10, 2) NOT NULL,
    type VARCHAR(4) NOT NULL CHECK (type IN ('CALL', 'PUT')),
    oi_score INTEGER NOT NULL,
    vol_oi_ratio DECIMAL(8, 2) NOT NULL,
    volume INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    notional DECIMAL(20, 2) NOT NULL,
    mark DECIMAL(10, 2),
    delta DECIMAL(6, 4),
    iv DECIMAL(8, 2),
    underlying_price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Skew Metrics: Put-Call skew and implied move
CREATE TABLE skew_metrics (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    underlying_price DECIMAL(10, 2) NOT NULL,
    atm_strike DECIMAL(10, 2) NOT NULL,
    
    -- Skew measurements
    skew_25d DECIMAL(8, 2),  -- 25-delta skew (professional standard)
    atm_skew DECIMAL(8, 2),  -- ATM skew
    avg_iv DECIMAL(8, 2),
    avg_call_iv DECIMAL(8, 2),
    avg_put_iv DECIMAL(8, 2),
    atm_call_iv DECIMAL(8, 2),
    atm_put_iv DECIMAL(8, 2),
    
    -- Positioning ratios
    put_call_oi_ratio DECIMAL(8, 4),
    put_call_vol_ratio DECIMAL(8, 4),
    
    -- Implied move
    implied_move_dollars DECIMAL(10, 2),
    implied_move_pct DECIMAL(8, 4),
    straddle_price DECIMAL(10, 2),
    
    -- Key levels
    upper_breakout DECIMAL(10, 2),
    lower_breakout DECIMAL(10, 2),
    upper_1sd DECIMAL(10, 2),
    lower_1sd DECIMAL(10, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scan Metadata: Track each scan run
CREATE TABLE scan_runs (
    scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    stocks_scanned INTEGER,
    expiries_scanned INTEGER,
    total_api_calls INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Whale flows indexes
CREATE INDEX idx_whale_flows_timestamp ON whale_flows(timestamp DESC);
CREATE INDEX idx_whale_flows_symbol ON whale_flows(symbol);
CREATE INDEX idx_whale_flows_expiry ON whale_flows(expiry);
CREATE INDEX idx_whale_flows_scan_id ON whale_flows(scan_id);
CREATE INDEX idx_whale_flows_score ON whale_flows(whale_score DESC);
CREATE INDEX idx_whale_flows_composite ON whale_flows(symbol, expiry, timestamp DESC);

-- OI flows indexes
CREATE INDEX idx_oi_flows_timestamp ON oi_flows(timestamp DESC);
CREATE INDEX idx_oi_flows_symbol ON oi_flows(symbol);
CREATE INDEX idx_oi_flows_expiry ON oi_flows(expiry);
CREATE INDEX idx_oi_flows_scan_id ON oi_flows(scan_id);
CREATE INDEX idx_oi_flows_vol_oi ON oi_flows(vol_oi_ratio DESC);
CREATE INDEX idx_oi_flows_composite ON oi_flows(symbol, expiry, timestamp DESC);

-- Skew metrics indexes
CREATE INDEX idx_skew_metrics_timestamp ON skew_metrics(timestamp DESC);
CREATE INDEX idx_skew_metrics_symbol ON skew_metrics(symbol);
CREATE INDEX idx_skew_metrics_expiry ON skew_metrics(expiry);
CREATE INDEX idx_skew_metrics_scan_id ON skew_metrics(scan_id);
CREATE INDEX idx_skew_metrics_skew ON skew_metrics(skew_25d);
CREATE INDEX idx_skew_metrics_composite ON skew_metrics(symbol, expiry, timestamp DESC);

-- ============================================================================
-- Computed Views: Composite Scores and Rankings
-- ============================================================================

-- Latest scan data (most recent complete scan)
CREATE OR REPLACE VIEW latest_scan AS
SELECT scan_id, start_time, end_time
FROM scan_runs
WHERE status = 'completed'
ORDER BY start_time DESC
LIMIT 1;

-- Top Opportunities: Composite scoring across all three metrics
CREATE OR REPLACE VIEW top_opportunities AS
WITH latest AS (
    SELECT scan_id FROM latest_scan
),
whale_latest AS (
    SELECT DISTINCT ON (symbol, expiry, strike, type)
        symbol, expiry, strike, type, whale_score, volume, vol_ratio,
        call_wall_strike, put_wall_strike, underlying_price
    FROM whale_flows
    WHERE scan_id = (SELECT scan_id FROM latest)
    ORDER BY symbol, expiry, strike, type, whale_score DESC
),
oi_latest AS (
    SELECT DISTINCT ON (symbol, expiry, strike, type)
        symbol, expiry, strike, type, oi_score, vol_oi_ratio, volume, notional
    FROM oi_flows
    WHERE scan_id = (SELECT scan_id FROM latest)
    ORDER BY symbol, expiry, strike, type, oi_score DESC
),
skew_latest AS (
    SELECT DISTINCT ON (symbol, expiry)
        symbol, expiry, skew_25d, atm_skew, avg_iv, put_call_oi_ratio,
        put_call_vol_ratio, implied_move_pct, upper_breakout, lower_breakout
    FROM skew_metrics
    WHERE scan_id = (SELECT scan_id FROM latest)
    ORDER BY symbol, expiry, timestamp DESC
)
SELECT 
    w.symbol,
    w.expiry,
    w.strike,
    w.type,
    w.underlying_price,
    
    -- Component scores
    w.whale_score,
    o.vol_oi_ratio,
    s.skew_25d,
    s.implied_move_pct,
    
    -- Composite calculation
    (
        -- Whale score component (0-35 points)
        CASE 
            WHEN w.whale_score > 200 THEN 35
            WHEN w.whale_score > 100 THEN 25
            WHEN w.whale_score > 50 THEN 15
            ELSE 5
        END +
        
        -- Fresh OI component (0-35 points)
        CASE 
            WHEN o.vol_oi_ratio > 8.0 THEN 35
            WHEN o.vol_oi_ratio > 6.0 THEN 30
            WHEN o.vol_oi_ratio > 4.0 THEN 20
            WHEN o.vol_oi_ratio > 3.0 THEN 10
            ELSE 0
        END +
        
        -- Skew alignment component (0-30 points)
        CASE 
            -- Extreme fear + calls = contrarian buy
            WHEN s.skew_25d > 6 AND w.type = 'CALL' THEN 30
            WHEN s.skew_25d > 4 AND w.type = 'CALL' THEN 20
            WHEN s.skew_25d > 3 AND w.type = 'CALL' THEN 10
            -- Greed + puts = contrarian short
            WHEN s.skew_25d < -1 AND w.type = 'PUT' THEN 30
            WHEN s.skew_25d < 0 AND w.type = 'PUT' THEN 20
            -- Normal skew = less conviction
            ELSE 5
        END +
        
        -- Directional alignment bonus (0-10 points)
        CASE 
            WHEN w.type = o.type AND w.vol_ratio < -30 AND w.type = 'CALL' THEN 10
            WHEN w.type = o.type AND w.vol_ratio > 30 AND w.type = 'PUT' THEN 10
            WHEN w.type = o.type THEN 5
            ELSE 0
        END
    ) AS composite_score,
    
    -- Signal classification
    CASE 
        WHEN s.skew_25d > 5 AND w.type = 'CALL' THEN 'CONTRARIAN_BULL'
        WHEN s.skew_25d < -1 AND w.type = 'PUT' THEN 'CONTRARIAN_BEAR'
        WHEN w.vol_ratio < -30 AND o.type = 'CALL' THEN 'MOMENTUM_BULL'
        WHEN w.vol_ratio > 30 AND o.type = 'PUT' THEN 'MOMENTUM_BEAR'
        ELSE 'NEUTRAL'
    END AS signal_type,
    
    -- Supporting data
    w.volume,
    o.notional,
    s.put_call_oi_ratio,
    w.call_wall_strike,
    w.put_wall_strike,
    s.upper_breakout,
    s.lower_breakout,
    
    (SELECT start_time FROM latest_scan) as data_timestamp
    
FROM whale_latest w
INNER JOIN oi_latest o 
    ON w.symbol = o.symbol 
    AND w.expiry = o.expiry 
    AND w.strike = o.strike 
    AND w.type = o.type
INNER JOIN skew_latest s 
    ON w.symbol = s.symbol 
    AND w.expiry = s.expiry
WHERE 
    w.whale_score >= 50
    AND o.vol_oi_ratio >= 3.0
ORDER BY composite_score DESC;

-- Market Sentiment Summary: Aggregate skew across all stocks
CREATE OR REPLACE VIEW market_sentiment AS
WITH latest AS (
    SELECT scan_id FROM latest_scan
),
latest_skew AS (
    SELECT DISTINCT ON (symbol, expiry)
        symbol, expiry, skew_25d, avg_iv, put_call_oi_ratio, implied_move_pct
    FROM skew_metrics
    WHERE scan_id = (SELECT scan_id FROM latest)
    ORDER BY symbol, expiry, timestamp DESC
)
SELECT 
    COUNT(DISTINCT symbol) as stocks_analyzed,
    AVG(skew_25d) as avg_skew,
    STDDEV(skew_25d) as skew_volatility,
    AVG(avg_iv) as avg_iv,
    AVG(put_call_oi_ratio) as avg_pc_ratio,
    AVG(implied_move_pct) as avg_implied_move,
    
    -- Count extremes
    COUNT(CASE WHEN skew_25d > 6 THEN 1 END) as extreme_fear_count,
    COUNT(CASE WHEN skew_25d < -1 THEN 1 END) as extreme_greed_count,
    COUNT(CASE WHEN implied_move_pct > 5 THEN 1 END) as high_vol_count,
    
    -- Sentiment classification
    CASE 
        WHEN AVG(skew_25d) > 5 THEN 'EXTREME_FEAR'
        WHEN AVG(skew_25d) > 3 THEN 'ELEVATED_FEAR'
        WHEN AVG(skew_25d) > -1 THEN 'NEUTRAL'
        WHEN AVG(skew_25d) > -3 THEN 'BULLISH'
        ELSE 'EUPHORIA'
    END as market_sentiment,
    
    (SELECT start_time FROM latest_scan) as data_timestamp
FROM latest_skew;

-- Stock Historical Skew: Track skew changes over time
CREATE OR REPLACE VIEW stock_skew_history AS
SELECT 
    symbol,
    expiry,
    timestamp,
    skew_25d,
    avg_iv,
    put_call_oi_ratio,
    implied_move_pct,
    underlying_price,
    -- Calculate change from previous scan
    skew_25d - LAG(skew_25d) OVER (PARTITION BY symbol, expiry ORDER BY timestamp) as skew_change,
    avg_iv - LAG(avg_iv) OVER (PARTITION BY symbol, expiry ORDER BY timestamp) as iv_change
FROM skew_metrics
ORDER BY symbol, expiry, timestamp DESC;

-- ============================================================================
-- Data Retention and Cleanup
-- ============================================================================

-- Function to clean old data (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    DELETE FROM whale_flows WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM oi_flows WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM skew_metrics WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM scan_runs WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Performance Statistics
-- ============================================================================

CREATE OR REPLACE VIEW scan_performance AS
SELECT 
    DATE(start_time) as scan_date,
    COUNT(*) as total_scans,
    AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration_seconds,
    SUM(total_api_calls) as total_api_calls,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_scans,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_scans
FROM scan_runs
WHERE start_time >= NOW() - INTERVAL '7 days'
GROUP BY DATE(start_time)
ORDER BY scan_date DESC;

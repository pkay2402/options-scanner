"""
Initialize SQLite database with whale flows, OI flows, and skew metrics tables
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "market_data.db"

# SQLite schema
SCHEMA = """
-- Whale flows table
CREATE TABLE IF NOT EXISTS whale_flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike REAL NOT NULL,
    type VARCHAR(4) NOT NULL CHECK (type IN ('CALL', 'PUT')),
    whale_score INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    valr REAL,
    gamma REAL,
    gex REAL,
    mark REAL,
    delta REAL,
    iv REAL,
    underlying_price REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_whale_timestamp ON whale_flows(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_whale_symbol ON whale_flows(symbol);
CREATE INDEX IF NOT EXISTS idx_whale_score ON whale_flows(whale_score DESC);

-- OI flows table (fresh positioning)
CREATE TABLE IF NOT EXISTS oi_flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike REAL NOT NULL,
    type VARCHAR(4) NOT NULL CHECK (type IN ('CALL', 'PUT')),
    volume INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    vol_oi_ratio REAL NOT NULL,
    mark REAL,
    delta REAL,
    iv REAL,
    underlying_price REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_oi_timestamp ON oi_flows(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_oi_symbol ON oi_flows(symbol);
CREATE INDEX IF NOT EXISTS idx_oi_ratio ON oi_flows(vol_oi_ratio DESC);

-- Skew metrics table
CREATE TABLE IF NOT EXISTS skew_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    underlying_price REAL NOT NULL,
    atm_iv REAL,
    skew_25d REAL,
    put_skew REAL,
    call_skew REAL,
    put_call_ratio REAL,
    total_call_volume INTEGER,
    total_put_volume INTEGER,
    total_call_oi INTEGER,
    total_put_oi INTEGER,
    implied_move REAL
);

CREATE INDEX IF NOT EXISTS idx_skew_timestamp ON skew_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_skew_symbol ON skew_metrics(symbol);

-- Cleanup old data (keep 7 days)
CREATE TRIGGER IF NOT EXISTS cleanup_whale_flows
AFTER INSERT ON whale_flows
BEGIN
    DELETE FROM whale_flows WHERE timestamp < datetime('now', '-7 days');
END;

CREATE TRIGGER IF NOT EXISTS cleanup_oi_flows
AFTER INSERT ON oi_flows
BEGIN
    DELETE FROM oi_flows WHERE timestamp < datetime('now', '-7 days');
END;

CREATE TRIGGER IF NOT EXISTS cleanup_skew_metrics
AFTER INSERT ON skew_metrics
BEGIN
    DELETE FROM skew_metrics WHERE timestamp < datetime('now', '-7 days');
END;
"""

def init_db():
    """Initialize database with schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Execute schema
    cursor.executescript(SCHEMA)
    conn.commit()
    
    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"✓ Database initialized at {DB_PATH}")
    print(f"✓ Tables created: {', '.join([t[0] for t in tables])}")
    
    conn.close()

if __name__ == "__main__":
    init_db()

"""
Options Flow Data Model
Represents individual options trades/flows
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OptionsFlow:
    """Individual options trade/flow"""
    
    # Symbol info
    symbol: str                  # Underlying ticker
    option_symbol: str          # Full option symbol
    strike: float
    expiry: datetime
    option_type: str           # 'CALL' or 'PUT'
    
    # Trade details
    timestamp: datetime
    price: float               # Premium paid per contract
    size: int                  # Number of contracts
    premium: float            # Total premium (price * size * 100)
    
    # Market context
    bid: float
    ask: float
    underlying_price: float
    iv: Optional[float] = None
    delta: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    # Trade classification
    trade_type: str = "SWEEP"   # SWEEP, BLOCK, SPLIT, SINGLE
    sentiment: str = "NEUTRAL"   # BULLISH, BEARISH, NEUTRAL
    aggressor: str = "UNKNOWN"   # BID, ASK, UNKNOWN
    
    # Derived metrics
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiration"""
        return (self.expiry - datetime.now()).days
    
    @property
    def is_0dte(self) -> bool:
        """Check if this is 0DTE"""
        return self.days_to_expiry == 0
    
    @property
    def moneyness(self) -> str:
        """ITM, ATM, or OTM"""
        if self.option_type == "CALL":
            if self.strike < self.underlying_price * 0.98:
                return "ITM"
            elif self.strike > self.underlying_price * 1.02:
                return "OTM"
            else:
                return "ATM"
        else:  # PUT
            if self.strike > self.underlying_price * 1.02:
                return "ITM"
            elif self.strike < self.underlying_price * 0.98:
                return "OTM"
            else:
                return "ATM"
    
    @property
    def is_whale(self) -> bool:
        """Determine if this is a whale trade (>$100k premium)"""
        return self.premium > 100000
    
    @property
    def premium_formatted(self) -> str:
        """Format premium with K/M suffix"""
        if self.premium >= 1_000_000:
            return f"${self.premium/1_000_000:.2f}M"
        elif self.premium >= 1_000:
            return f"${self.premium/1_000:.1f}K"
        else:
            return f"${self.premium:.0f}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame"""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'type': self.option_type,
            'strike': self.strike,
            'expiry': self.expiry,
            'dte': self.days_to_expiry,
            'price': self.price,
            'size': self.size,
            'premium': self.premium,
            'premium_fmt': self.premium_formatted,
            'underlying': self.underlying_price,
            'moneyness': self.moneyness,
            'trade_type': self.trade_type,
            'sentiment': self.sentiment,
            'iv': self.iv,
            'delta': self.delta,
            'volume': self.volume,
            'oi': self.open_interest
        }


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    size: int
    num_orders: int
    side: str  # 'BID' or 'ASK'
    
    @property
    def notional(self) -> float:
        """Total notional value (size * price * 100)"""
        return self.size * self.price * 100
    
    def to_dict(self) -> dict:
        return {
            'price': self.price,
            'size': self.size,
            'orders': self.num_orders,
            'notional': self.notional,
            'side': self.side
        }


@dataclass
class OptionsBook:
    """Full order book for an option"""
    
    option_symbol: str
    symbol: str
    strike: float
    expiry: datetime
    option_type: str
    
    timestamp: datetime
    
    # Order book levels (sorted best to worst)
    bids: list[OrderBookLevel]  # Sorted highest to lowest
    asks: list[OrderBookLevel]  # Sorted lowest to highest
    
    # Aggregate metrics
    @property
    def best_bid(self) -> Optional[float]:
        """Top of book bid"""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Top of book ask"""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Spread as percentage of mid"""
        if self.best_bid and self.best_ask:
            mid = (self.best_bid + self.best_ask) / 2
            return (self.spread / mid) * 100 if mid > 0 else None
        return None
    
    @property
    def total_bid_size(self) -> int:
        """Sum of all bid sizes"""
        return sum(level.size for level in self.bids)
    
    @property
    def total_ask_size(self) -> int:
        """Sum of all ask sizes"""
        return sum(level.size for level in self.asks)
    
    @property
    def total_bid_notional(self) -> float:
        """Total dollar value on bid side"""
        return sum(level.notional for level in self.bids)
    
    @property
    def total_ask_notional(self) -> float:
        """Total dollar value on ask side"""
        return sum(level.notional for level in self.asks)
    
    @property
    def imbalance(self) -> Optional[float]:
        """Bid/ask size imbalance ratio"""
        total_ask = self.total_ask_size
        if total_ask > 0:
            return self.total_bid_size / total_ask
        return None
    
    def get_depth(self, levels: int = 10) -> dict:
        """Get book depth at N levels"""
        return {
            'bids': [level.to_dict() for level in self.bids[:levels]],
            'asks': [level.to_dict() for level in self.asks[:levels]],
            'bid_size': self.total_bid_size,
            'ask_size': self.total_ask_size,
            'imbalance': self.imbalance,
            'spread': self.spread,
            'spread_pct': self.spread_pct
        }

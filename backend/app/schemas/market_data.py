from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class MarketDataBase(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    rsi: Optional[float] = None
    macd: Optional[float] = None
    signal_line: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    atr: Optional[float] = None

    class Config:
        from_attributes = True

class MarketDataResponse(MarketDataBase):
    pass

class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_close: float
    confidence: float
    trend: str  # "up" or "down"
    supporting_factors: List[str]

    class Config:
        from_attributes = True 
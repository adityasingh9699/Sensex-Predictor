from sqlalchemy import Column, Integer, Float, String, Date, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class HistoricalData(Base):
    __tablename__ = "historical_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Technical Indicators
    rsi = Column(Float)
    macd = Column(Float)
    signal_line = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    atr = Column(Float)
    
    # Sentiment Score from News
    sentiment_score = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (UniqueConstraint('date', name='unique_date_historical'),)

class DailyData(Base):
    __tablename__ = "daily_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Technical Indicators
    rsi = Column(Float)
    macd = Column(Float)
    signal_line = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    atr = Column(Float)
    
    # Predictions
    predicted_close = Column(Float)
    trend = Column(String(10))
    confidence = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (UniqueConstraint('timestamp', name='unique_timestamp_daily'),)

class NewsData(Base):
    __tablename__ = "news_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(String(5000))
    source = Column(String(100))
    sentiment_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date, timedelta
from typing import Optional, List
from pydantic import BaseModel
import logging
import asyncio
import pytz
import pandas as pd

from app.config.settings import settings
from app.database.database import init_db
from app.services.scheduler_service import SchedulerService
from app.services.model_service import ModelService
from app.services.data_service import DataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class Prediction(BaseModel):
    predicted_price: float
    actual_price: float
    trend: str
    confidence: float
    last_close: float
    timestamp: str
    price_change_pct: float
    metrics: dict

class MarketData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    rsi: Optional[float]
    macd: Optional[float]
    signal_line: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_middle: Optional[float]
    atr: Optional[float]
    sentiment_score: Optional[float]

# Service instances
model_service = None
scheduler_service = None
data_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        global data_service, model_service, scheduler_service
        
        # Initialize database tables
        init_db()
        
        # Initialize data service
        data_service = DataService()
        
        # Initialize model service
        model_service = ModelService(data_service)
        
        # Initialize and start scheduler
        scheduler_service = SchedulerService(data_service, model_service)
        await scheduler_service.start()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if scheduler_service:
            await scheduler_service.stop()
        if model_service:
            await model_service.stop_training()
        logger.info("Successfully shut down all services")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": f"{settings.PROJECT_NAME} API is running"}

@app.get(f"{settings.API_V1_STR}/predict", response_model=Prediction)
async def get_prediction():
    """Get latest market prediction"""
    try:
        if not model_service:
            raise HTTPException(status_code=503, detail="Model service not available")
        
        prediction = model_service.predict()  # Remove await since predict is synchronous
        if not prediction:
            raise HTTPException(status_code=404, detail="Could not generate prediction")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/historical-data", response_model=List[MarketData])
@app.get(f"{settings.API_V1_STR}/market-data", response_model=List[MarketData])
async def get_historical_data():
    """Get historical market data"""
    try:
        if not data_service:
            raise HTTPException(status_code=503, detail="Data service not available")
        
        # Get data for last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # First try to fetch data, if empty, initialize it
        df = data_service.get_historical_data_range(
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            logger.info("No historical data found, fetching from Yahoo Finance...")
            await data_service.get_historical_data()  # This will fetch and store data
            
            # Try fetching again
            df = data_service.get_historical_data_range(
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                raise HTTPException(status_code=404, detail="Could not fetch historical data")
        
        # Convert DataFrame to list of MarketData
        data = []
        for idx, row in df.iterrows():
            data_point = {
                'timestamp': idx,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']),
                'rsi': float(row['rsi']) if pd.notnull(row['rsi']) else None,
                'macd': float(row['macd']) if pd.notnull(row['macd']) else None,
                'signal_line': float(row['signal_line']) if pd.notnull(row['signal_line']) else None,
                'bb_upper': float(row['bb_upper']) if pd.notnull(row['bb_upper']) else None,
                'bb_lower': float(row['bb_lower']) if pd.notnull(row['bb_lower']) else None,
                'bb_middle': float(row['bb_middle']) if pd.notnull(row['bb_middle']) else None,
                'atr': float(row['atr']) if pd.notnull(row['atr']) else None,
                'sentiment_score': float(row['sentiment_score']) if pd.notnull(row['sentiment_score']) else None
            }
            data.append(MarketData(**data_point))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/live-data", response_model=List[MarketData])
async def get_live_data():
    """Get today's live market data"""
    try:
        if not data_service:
            raise HTTPException(status_code=503, detail="Data service not available")
        
        df = data_service.get_live_data()
        if df.empty:
            raise HTTPException(status_code=404, detail="No live data found")
        
        # Convert DataFrame to list of MarketData
        data = []
        for idx, row in df.iterrows():
            data_point = {
                'timestamp': idx,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']),
                'rsi': float(row['rsi']) if pd.notnull(row['rsi']) else None,
                'macd': float(row['macd']) if pd.notnull(row['macd']) else None,
                'signal_line': float(row['signal_line']) if pd.notnull(row['signal_line']) else None,
                'bb_upper': float(row['bb_upper']) if pd.notnull(row['bb_upper']) else None,
                'bb_lower': float(row['bb_lower']) if pd.notnull(row['bb_lower']) else None,
                'bb_middle': float(row['bb_middle']) if pd.notnull(row['bb_middle']) else None,
                'atr': float(row['atr']) if pd.notnull(row['atr']) else None,
                'sentiment_score': float(row['sentiment_score']) if pd.notnull(row['sentiment_score']) else None
            }
            data.append(MarketData(**data_point))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting live data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/market-summary")
async def get_market_summary():
    """Get market summary with prediction and current metrics"""
    try:
        if not data_service or not model_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get live data
        live_data = data_service.get_live_data()
        if not live_data:
            raise HTTPException(status_code=404, detail="No live data found")
        
        # Get prediction
        prediction = await model_service.predict()
        if not prediction:
            raise HTTPException(status_code=404, detail="Could not generate prediction")
        
        # Calculate metrics
        current_price = live_data[-1]['close'] if live_data else None
        prev_close = live_data[-2]['close'] if len(live_data) > 1 else None
        
        daily_change = None
        daily_change_percent = None
        if current_price and prev_close:
            daily_change = current_price - prev_close
            daily_change_percent = (daily_change / prev_close) * 100
        
        return {
            "current_price": current_price,
            "daily_change": daily_change,
            "daily_change_percent": daily_change_percent,
            "prediction": prediction,
            "last_updated": datetime.now(pytz.timezone(settings.TIMEZONE)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 
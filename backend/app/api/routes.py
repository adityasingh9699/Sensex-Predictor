from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from app.database.database import get_db
from app.services.data_service import DataService
from app.services.model_service import ModelService
from app.schemas.market_data import MarketDataResponse, PredictionResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["market"],
    responses={404: {"description": "Not found"}},
)

@router.get("/predict", response_model=PredictionResponse)
async def get_prediction(db: Session = Depends(get_db)):
    """Get market prediction for a specific date"""
    try:
        model_service = ModelService(db)
        prediction = await model_service.predict()
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical-data", response_model=List[MarketDataResponse])
async def get_historical_data(db: Session = Depends(get_db)):
    """Get historical market data"""
    try:
        data_service = DataService(db)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        data = data_service.get_historical_data_range(start_date, end_date)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live-data", response_model=List[MarketDataResponse])
async def get_live_data(db: Session = Depends(get_db)):
    """Get today's live market data"""
    try:
        data_service = DataService(db)
        data = data_service.get_live_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
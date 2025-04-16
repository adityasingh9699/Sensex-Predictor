"""
Sensex Predictor API
-------------------
Main application package initialization.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database.database import init_db
import logging

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Sensex Predictor API",
        description="API for predicting Sensex market movements",
        version="1.0.0"
    )
    
    # Add CORS middleware with more permissive settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # More permissive for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    # Initialize database
    init_db()
    
    # Include API routes with tags
    app.include_router(
        router,
        tags=["market"],
        responses={404: {"description": "Not found"}},
    )
    
    # Log registered routes
    for route in app.routes:
        logger.info(f"Registered route: {route.path} [{route.methods}]")
    
    return app 
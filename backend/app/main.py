import logging
from app import create_app
from app.database.database import SessionLocal
from app.services.scheduler_service import SchedulerService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        db = SessionLocal()
        scheduler = SchedulerService(db)
        await scheduler.start()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
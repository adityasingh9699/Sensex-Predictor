import asyncio
import logging
from datetime import datetime, time
import pytz
from app.services.data_service import DataService
from app.services.model_service import ModelService
from app.config.settings import settings

logger = logging.getLogger(__name__)

class SchedulerService:
    def __init__(self, data_service: DataService, model_service: ModelService):
        self.data_service = data_service
        self.model_service = model_service
        self.is_running = False
        self.update_interval = 10  # 10 seconds interval
        self.tasks = []
        self.ist_tz = pytz.timezone('Asia/Kolkata')  # Indian Standard Time timezone

    async def start(self):
        """Start all scheduler tasks"""
        if self.is_running:
            return
            
        self.is_running = True
        try:
            # Initialize data on startup
            await self.initial_setup()
            
            # Create tasks for continuous updates
            self.tasks = [
                asyncio.create_task(self._run_continuous_updates()),
                asyncio.create_task(self.model_service.train_continuously())
            ]
            
            logger.info("Scheduler service started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {str(e)}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop all scheduler tasks"""
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.tasks = []
        logger.info("Scheduler service stopped")

    async def _run_continuous_updates(self):
        """Run continuous updates for data"""
        while self.is_running:
            try:
                now = datetime.now(self.ist_tz)
                
                # Only update during market hours on weekdays
                if self._is_market_hours(now):
                    # Update market data
                    self.data_service.update_live_data()
                    logger.info("Updated live market data")
                    
                    # Update news and sentiment
                    self.data_service.update_news_data()
                    logger.info("Updated news data")
                    
                    # Transfer daily data to historical if needed
                    self.data_service.transfer_daily_to_historical()
                else:
                    logger.debug("Skipping updates - outside market hours")
                
            except Exception as e:
                logger.error(f"Error in continuous updates: {str(e)}")
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)

    async def initial_setup(self):
        """Perform initial data setup"""
        try:
            # Fetch historical data if needed
            self.data_service.get_historical_data()
            
            # Get initial market data
            self.data_service.update_live_data()
            
            # Get initial news data
            self.data_service.update_news_data()
            
        except Exception as e:
            logger.error(f"Error in initial setup: {str(e)}")
            raise

    def _is_market_hours(self, current_time: datetime) -> bool:
        """Check if current time is during market hours"""
        # First check if it's a weekday (Monday = 0, Friday = 4)
        if current_time.weekday() >= 5:  # Weekend
            return False
            
        # Then check if it's during market hours (9:15 AM to 3:30 PM)
        market_time = current_time.time()
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        
        return market_open <= market_time <= market_close

    async def schedule_data_updates(self):
        """Schedule regular data updates"""
        while self.is_running:
            try:
                now = datetime.now(self.ist_tz)
                
                # Check if it's a weekday
                if now.weekday() < 5:  # Monday = 0, Friday = 4
                    # Update market data every 5 minutes during market hours
                    if self._is_market_hours(now):
                        self.data_service.update_live_data()
                        await asyncio.sleep(300)  # 5 minutes
                    else:
                        # Transfer daily data to historical after market hours
                        self.data_service.transfer_daily_to_historical()
                        await asyncio.sleep(3600)  # 1 hour
                        
                    # Update news every hour
                    if now.minute < 5:  # Update at the start of each hour
                        self.data_service.update_news_data()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying 
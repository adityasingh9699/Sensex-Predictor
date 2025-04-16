import pymysql
from pymysql.cursors import DictCursor
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_db():
    """Get database connection"""
    try:
        connection = pymysql.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            port=settings.DB_PORT,
            cursorclass=DictCursor,  # Returns results as dictionaries
            charset='utf8mb4'
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables"""
    connection = get_db()
    try:
        with connection.cursor() as cursor:
            # Create historical_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    open FLOAT NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    volume BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY date_idx (date)
                )
            """)
            
            # Create daily_updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_updates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    price FLOAT NOT NULL,
                    volume BIGINT NOT NULL,
                    sentiment FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY date_time_idx (date, time)
                )
            """)
            
            # Create news_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    source VARCHAR(255),
                    sentiment FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX date_idx (date)
                )
            """)
            
        connection.commit()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        connection.close() 
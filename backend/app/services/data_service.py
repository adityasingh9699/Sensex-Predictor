import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from newsapi import NewsApiClient
from transformers import pipeline
import ta
from app.database.database import get_db
from app.config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        """Initialize the DataService without requiring a db connection"""
        # BSE SENSEX symbols (try multiple in case one fails)
        self.symbols = [
            "^BSESN",  # Yahoo Finance
            "SENSEX.BO",  # Alternative Yahoo Finance
            "BSE.NS",   # NSE equivalent
            "BSE:SENSEX"  # Another format
        ]
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=0 if settings.USE_GPU else -1
        )
        try:
            self.news_api = NewsApiClient(api_key=settings.NEWS_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize NewsApiClient: {str(e)}")
            self.news_api = None
        self.ist_tz = pytz.timezone(settings.TIMEZONE)
    
    def get_db_connection(self):
        """Get a new database connection"""
        return get_db()
    
    def check_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.ist_tz)
        market_close = now.replace(
            hour=settings.MARKET_CLOSE_TIME.hour,
            minute=settings.MARKET_CLOSE_TIME.minute
        )
        return now <= market_close
    
    async def get_historical_data(self) -> bool:
        """Fetch and store historical data if not present"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                # Check if we have sufficient historical data
                cursor.execute("""
                    SELECT MIN(date) as earliest_date
                    FROM historical_data
                """)
                result = cursor.fetchone()
                earliest_date = result['earliest_date'] if result else None
                
                if earliest_date:
                    years_of_data = (datetime.now().date() - earliest_date).days / 365
                    if years_of_data >= settings.YEARS_OF_HISTORICAL_DATA:
                        return False
            
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=settings.YEARS_OF_HISTORICAL_DATA*365)
            
            sensex = yf.Ticker(self.symbol)
            df = sensex.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning("No historical data fetched")
                return False
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Store in database
            with conn.cursor() as cursor:
                for index, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO historical_data 
                        (date, open, high, low, close, volume, rsi, macd, signal_line, 
                        bb_upper, bb_lower, bb_middle, atr)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume),
                        rsi = VALUES(rsi),
                        macd = VALUES(macd),
                        signal_line = VALUES(signal_line),
                        bb_upper = VALUES(bb_upper),
                        bb_lower = VALUES(bb_lower),
                        bb_middle = VALUES(bb_middle),
                        atr = VALUES(atr)
                    """, (
                        index.date(),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['RSI']),
                        float(row['MACD']),
                        float(row['Signal_Line']),
                        float(row['BB_Upper']),
                        float(row['BB_Lower']),
                        float(row['BB_Middle']),
                        float(row['ATR'])
                    ))
            
            conn.commit()
            logger.info(f"Saved {len(df)} historical records")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    async def update_live_data(self):
        """Fetch and store live market data"""
        if not self.check_market_hours():
            return
            
        conn = None
        try:
            # Fetch latest data
            sensex = yf.Ticker(self.symbol)
            df = sensex.history(period="1d", interval="5m")
            
            if df.empty:
                return
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Store latest data point
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                latest = df.iloc[-1]
                cursor.execute("""
                    INSERT INTO daily_data (
                        timestamp, open, high, low, close, volume, 
                        rsi, macd, signal_line, bb_upper, bb_lower, 
                        bb_middle, atr
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume),
                        rsi = VALUES(rsi),
                        macd = VALUES(macd),
                        signal_line = VALUES(signal_line),
                        bb_upper = VALUES(bb_upper),
                        bb_lower = VALUES(bb_lower),
                        bb_middle = VALUES(bb_middle),
                        atr = VALUES(atr)
                """, (
                    df.index[-1],
                    float(latest['Open']),
                    float(latest['High']),
                    float(latest['Low']),
                    float(latest['Close']),
                    int(latest['Volume']),
                    float(latest['RSI']),
                    float(latest['MACD']),
                    float(latest['Signal_Line']),
                    float(latest['BB_Upper']),
                    float(latest['BB_Lower']),
                    float(latest['BB_Middle']),
                    float(latest['ATR'])
                ))
            
            conn.commit()
            logger.info("Updated live market data")
            
        except Exception as e:
            logger.error(f"Error updating live data: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        try:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['Signal_Line'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            
            # ATR
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close']
            ).average_true_range()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def _fetch_yf_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Try to fetch data from Yahoo Finance using multiple symbols"""
        for symbol in self.symbols:
            try:
                logger.info(f"Trying to fetch data with symbol: {symbol}")
                sensex = yf.Ticker(symbol)
                df = sensex.history(start=start_date, end=end_date)
                
                if not df.empty:
                    logger.info(f"Successfully fetched data using symbol: {symbol}")
                    return df
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data with symbol {symbol}: {str(e)}")
                continue
        
        logger.error("Failed to fetch data with all available symbols")
        return pd.DataFrame()

    def get_historical_data_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data within date range"""
        conn = None
        try:
            # If we don't have enough data, fetch it first
            min_required_days = settings.SEQUENCE_LENGTH + 10  # Add buffer
            if (end_date - start_date).days < min_required_days:
                start_date = end_date - timedelta(days=min_required_days * 2)  # Double the days to ensure enough data
            
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        date as timestamp,
                        open, high, low, close, volume,
                        rsi, macd, signal_line,
                        bb_upper, bb_lower, bb_middle, atr,
                        COALESCE(
                            (SELECT sentiment_score 
                             FROM news_data 
                             WHERE date = historical_data.date 
                             LIMIT 1),
                            0
                        ) as sentiment_score
                    FROM historical_data
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date ASC
                """, (start_date.date(), end_date.date()))
                
                results = cursor.fetchall()
                
                if not results or len(results) < min_required_days:
                    logger.info(f"Not enough data points in database, fetching from Yahoo Finance...")
                    
                    # Try to fetch data from Yahoo Finance
                    yf_df = self._fetch_yf_data(start_date, end_date)
                    
                    if not yf_df.empty:
                        # Calculate technical indicators
                        yf_df = self._calculate_technical_indicators(yf_df)
                        
                        # Store in database
                        self._store_historical_data(yf_df)
                        
                        # Retry database fetch with the new data
                        cursor.execute("""
                            SELECT 
                                date as timestamp,
                                open, high, low, close, volume,
                                rsi, macd, signal_line,
                                bb_upper, bb_lower, bb_middle, atr,
                                COALESCE(
                                    (SELECT sentiment_score 
                                     FROM news_data 
                                     WHERE date = historical_data.date 
                                     LIMIT 1),
                                    0
                                ) as sentiment_score
                            FROM historical_data
                            WHERE date BETWEEN %s AND %s
                            ORDER BY date ASC
                        """, (start_date.date(), end_date.date()))
                        
                        results = cursor.fetchall()
                
                if not results:
                    logger.warning("No historical data found for the specified date range")
                    return pd.DataFrame()
                
                # Convert results to DataFrame
                df = pd.DataFrame(results)
                df.set_index('timestamp', inplace=True)
                
                # Convert columns to proper types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                                 'signal_line', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 
                                 'sentiment_score']
                
                for col in numeric_columns:
                    if col in df.columns:
                        if col == 'volume':
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle missing values using modern methods
                df = df.ffill().bfill()
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching historical data range: {str(e)}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def _store_historical_data(self, df: pd.DataFrame):
        """Store historical data in database"""
        if df.empty:
            logger.warning("No data to store")
            return
            
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                stored_count = 0
                for index, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT INTO historical_data 
                            (date, open, high, low, close, volume, rsi, macd, signal_line, 
                            bb_upper, bb_lower, bb_middle, atr)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            open = VALUES(open),
                            high = VALUES(high),
                            low = VALUES(low),
                            close = VALUES(close),
                            volume = VALUES(volume),
                            rsi = VALUES(rsi),
                            macd = VALUES(macd),
                            signal_line = VALUES(signal_line),
                            bb_upper = VALUES(bb_upper),
                            bb_lower = VALUES(bb_lower),
                            bb_middle = VALUES(bb_middle),
                            atr = VALUES(atr)
                        """, (
                            index.date(),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Volume']),
                            float(row['RSI']),
                            float(row['MACD']),
                            float(row['Signal_Line']),
                            float(row['BB_Upper']),
                            float(row['BB_Lower']),
                            float(row['BB_Middle']),
                            float(row['ATR'])
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Error storing row for date {index.date()}: {str(e)}")
                        continue
                        
            conn.commit()
            logger.info(f"Successfully stored {stored_count} out of {len(df)} historical records")
            
        except Exception as e:
            logger.error(f"Error storing historical data: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def get_live_data(self) -> list:
        """Get today's live market data"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        timestamp,
                        open, high, low, close, volume,
                        rsi, macd, signal_line,
                        bb_upper, bb_lower, bb_middle, atr,
                        COALESCE(
                            (SELECT sentiment_score 
                             FROM news_data 
                             WHERE DATE(timestamp) = DATE(daily_data.timestamp) 
                             ORDER BY timestamp DESC 
                             LIMIT 1),
                            0
                        ) as sentiment_score
                    FROM daily_data
                    WHERE DATE(timestamp) = CURDATE()
                    ORDER BY timestamp ASC
                """)
                
                results = cursor.fetchall()
                
                # Convert to list of dicts with proper types
                data = []
                for row in results:
                    data_point = {
                        'timestamp': row['timestamp'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']),
                        'rsi': float(row['rsi']) if row['rsi'] is not None else None,
                        'macd': float(row['macd']) if row['macd'] is not None else None,
                        'signal_line': float(row['signal_line']) if row['signal_line'] is not None else None,
                        'bb_upper': float(row['bb_upper']) if row['bb_upper'] is not None else None,
                        'bb_lower': float(row['bb_lower']) if row['bb_lower'] is not None else None,
                        'bb_middle': float(row['bb_middle']) if row['bb_middle'] is not None else None,
                        'atr': float(row['atr']) if row['atr'] is not None else None,
                        'sentiment_score': float(row['sentiment_score']) if row['sentiment_score'] is not None else 0.0
                    }
                    data.append(data_point)
                
                return data
                
        except Exception as e:
            logger.error(f"Error fetching live data: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()

    async def transfer_daily_to_historical(self):
        """Transfer daily closing data to historical table"""
        if self.check_market_hours():
            return
            
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                # Get latest daily data
                cursor.execute("""
                    SELECT *
                    FROM daily_data
                    WHERE DATE(timestamp) = CURRENT_DATE()
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()
                
                if not latest:
                    return
                
                # Transfer to historical data
                cursor.execute("""
                    INSERT INTO historical_data (
                        date, open, high, low, close, volume,
                        rsi, macd, signal_line, bb_upper, bb_lower,
                        bb_middle, atr
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume),
                        rsi = VALUES(rsi),
                        macd = VALUES(macd),
                        signal_line = VALUES(signal_line),
                        bb_upper = VALUES(bb_upper),
                        bb_lower = VALUES(bb_lower),
                        bb_middle = VALUES(bb_middle),
                        atr = VALUES(atr)
                """, (
                    latest['timestamp'].date(),
                    latest['open'],
                    latest['high'],
                    latest['low'],
                    latest['close'],
                    latest['volume'],
                    latest['rsi'],
                    latest['macd'],
                    latest['signal_line'],
                    latest['bb_upper'],
                    latest['bb_lower'],
                    latest['bb_middle'],
                    latest['atr']
                ))
                
            conn.commit()
            logger.info("Transferred daily data to historical")
            
        except Exception as e:
            logger.error(f"Error transferring daily data: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    async def update_news_data(self):
        """Fetch and analyze news data"""
        if not self.news_api:
            logger.error("NewsApiClient not initialized")
            return
            
        conn = None
        try:
            # Fetch news
            news = self.news_api.get_everything(
                q="(Sensex OR BSE) AND India",
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            if not news.get('articles'):
                return
                
            conn = self.get_db_connection()
            with conn.cursor() as cursor:
                for article in news['articles']:
                    if not all([
                        article.get('title'),
                        article.get('description'),
                        article.get('publishedAt')
                    ]):
                        continue
                        
                    # Analyze sentiment
                    text = f"{article['title']}. {article['description']}"
                    sentiment = self.sentiment_analyzer(text)[0]
                    score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                    
                    # Store in database
                    cursor.execute("""
                        INSERT INTO news_data (
                            timestamp, title, content, source, sentiment_score
                        ) VALUES (
                            %s, %s, %s, %s, %s
                        ) ON DUPLICATE KEY UPDATE
                            content = VALUES(content),
                            sentiment_score = VALUES(sentiment_score)
                    """, (
                        article['publishedAt'],
                        article['title'],
                        article['description'],
                        article.get('source', {}).get('name', 'Unknown'),
                        score
                    ))
            
            conn.commit()
            logger.info(f"Processed {len(news['articles'])} news articles")
            
        except Exception as e:
            logger.error(f"Error updating news data: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using Hugging Face model"""
        try:
            result = self.sentiment_analyzer(text)[0]
            # Convert sentiment to numerical score (-1 to 1)
            if result['label'] == 'POSITIVE':
                return result['score']
            elif result['label'] == 'NEGATIVE':
                return -result['score']
            else:  # NEUTRAL
                return 0.0
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0
    
    def _get_source_impact(self, source: str) -> float:
        """Get impact multiplier based on source reliability"""
        source_impact = {
            'reuters.com': 1.0,
            'bloomberg.com': 1.0,
            'economictimes.indiatimes.com': 0.9,
            'moneycontrol.com': 0.9,
            'livemint.com': 0.8,
            'business-standard.com': 0.8,
            'thehindubusinessline.com': 0.8
        }
        return source_impact.get(source.lower(), 0.5)  # Default impact for unknown sources
    
    def get_news_sentiment(self, date: datetime) -> float:
        """Get average sentiment score for a specific date"""
        try:
            query = text("""
                SELECT AVG(sentiment_score) as avg_sentiment
                FROM news_data
                WHERE DATE(timestamp) = :date
            """)
            result = self.db.execute(query, {'date': date.date()}).fetchone()
            return float(result[0]) if result[0] is not None else 0.0
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return 0.0
    
    def get_latest_market_data(self):
        """Get the latest market data point"""
        try:
            # Try to get the latest daily data first
            latest = self.db.query(DailyData)\
                .order_by(DailyData.timestamp.desc())\
                .first()
            
            if latest:
                return latest

            # If no daily data, get the latest historical data
            return self.db.query(HistoricalData)\
                .order_by(HistoricalData.date.desc())\
                .first()
        except Exception as e:
            logger.error(f"Error getting latest market data: {str(e)}")
            return None

    def save_historical_data(self, data: pd.DataFrame):
        """Save historical data to database"""
        connection = get_db()
        try:
            with connection.cursor() as cursor:
                for _, row in data.iterrows():
                    cursor.execute("""
                        INSERT INTO historical_data (date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume)
                    """, (
                        row['date'].date(),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume'])
                    ))
            connection.commit()
            self.logger.info(f"Saved {len(data)} historical records")
        except Exception as e:
            self.logger.error(f"Error saving historical data: {str(e)}")
            raise
        finally:
            connection.close()

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data from database"""
        connection = get_db()
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM historical_data
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date
                """, (start_date, end_date))
                
                results = cursor.fetchall()
                
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
        finally:
            connection.close()

    def save_daily_update(self, date: datetime, price: float, volume: int, sentiment: float = None):
        """Save daily market update"""
        connection = get_db()
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO daily_updates (date, time, price, volume, sentiment)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    price = VALUES(price),
                    volume = VALUES(volume),
                    sentiment = VALUES(sentiment)
                """, (
                    date.date(),
                    date.time(),
                    price,
                    volume,
                    sentiment
                ))
            connection.commit()
            self.logger.info(f"Saved daily update for {date}")
        except Exception as e:
            self.logger.error(f"Error saving daily update: {str(e)}")
            raise
        finally:
            connection.close()

    def get_latest_updates(self, limit: int = 10) -> pd.DataFrame:
        """Get latest market updates"""
        connection = get_db()
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT date, time, price, volume, sentiment
                    FROM daily_updates
                    ORDER BY date DESC, time DESC
                    LIMIT %s
                """, (limit,))
                
                results = cursor.fetchall()
                
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching latest updates: {str(e)}")
            raise
        finally:
            connection.close() 
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:8000/api/v1';

interface Prediction {
  date: string;  // Changed from timestamp to match backend
  predicted_price: number;  // Changed from predicted_close to match backend
  actual_price: number | null;
  trend: 'up' | 'down' | 'neutral';
  confidence: number;
}

interface HistoricalData {
  date: string;  // Changed from timestamp to match backend
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

function App() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [predictionRes, historicalRes] = await Promise.all([
        axios.get<Prediction>(`${API_BASE_URL}/predict`, {
          timeout: 10000,
        }),
        axios.get<HistoricalData[]>(`${API_BASE_URL}/market-data`, {  // Changed from historical-data to market-data
          timeout: 10000,
        })
      ]);

      if (predictionRes.data === null) {
        throw new Error('Prediction data is not available');
      }

      setPrediction(predictionRes.data);
      setHistoricalData(historicalRes.data);
      setRetryCount(0);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      console.error('Error fetching data:', err);
      
      if (retryCount < 3) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          fetchData();
        }, 2000 * (retryCount + 1));
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRetry = () => {
    setRetryCount(0);
    fetchData();
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading market data...</p>
          {retryCount > 0 && (
            <p className="retry-message">
              Retry attempt {retryCount}/3...
            </p>
          )}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="App">
        <div className="error-container">
          <h2>Error Loading Data</h2>
          <p>{error}</p>
          <button onClick={handleRetry} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sensex Predictor</h1>
      </header>
      
      <main>
        {prediction && (
          <div className="prediction-section">
            <h2>Market Prediction</h2>
            <div className="metrics">
              <div className="metric">
                <h3>Predicted Price</h3>
                <p>₹{prediction.predicted_price?.toLocaleString() ?? 'N/A'}</p>
              </div>
              <div className="metric">
                <h3>Actual Price</h3>
                <p>₹{prediction.actual_price?.toLocaleString() ?? 'N/A'}</p>
              </div>
              <div className="metric">
                <h3>Trend</h3>
                <p className={prediction.trend}>{prediction.trend?.toUpperCase() ?? 'N/A'}</p>
              </div>
              <div className="metric">
                <h3>Confidence</h3>
                <p>{prediction.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : 'N/A'}</p>
              </div>
            </div>
          </div>
        )}

        {historicalData.length > 0 && (
          <div className="chart-section">
            <h2>Historical Data</h2>
            <LineChart width={800} height={400} data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="close" 
                stroke="#8884d8" 
                name="Close Price"
                dot={false}
              />
            </LineChart>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 
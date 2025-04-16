import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  useTheme,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';
import { getPrediction, getMarketData, getLiveData } from '../services/api';
import type { Prediction, MarketData } from '../services/api';

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [liveData, setLiveData] = useState<MarketData[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date | null>(new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        setError(null);
        const [predictionData, marketHistoryData, currentLiveData] = await Promise.all([
          getPrediction(),
          getMarketData(),
          getLiveData(),
        ]);
        setPrediction(predictionData);
        setMarketData(marketHistoryData);
        setLiveData(currentLiveData);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to fetch market data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
    const interval = setInterval(fetchInitialData, 5 * 60 * 1000); // Update every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const handleDateChange = async (date: Date | null) => {
    if (!date) return;
    setSelectedDate(date);
    try {
      setLoading(true);
      setError(null);
      const daysAgo = Math.floor(
        (new Date().getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
      );
      const predictionData = await getPrediction(daysAgo);
      setPrediction(predictionData);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError('Failed to fetch prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !marketData.length) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Typography variant="h1" gutterBottom>
        Sensex Predictor
      </Typography>

      {error && (
        <Typography color="error" gutterBottom>
          {error}
        </Typography>
      )}

      <Grid container spacing={3}>
        {/* Prediction Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h2" gutterBottom>
                Prediction
              </Typography>
              <DatePicker
                label="Select Date"
                value={selectedDate}
                onChange={handleDateChange}
                sx={{ mb: 2, width: '100%' }}
              />
              {prediction && (
                <Box mt={2}>
                  <Typography variant="h4" color={prediction.trend === 'BULLISH' ? 'success.main' : 'error.main'}>
                    {prediction.trend}
                  </Typography>
                  <Typography variant="body1" mt={1}>
                    Predicted: ₹{prediction.predicted_price.toLocaleString()}
                  </Typography>
                  <Typography variant="body1" mt={1}>
                    Confidence: {(prediction.confidence * 100).toFixed(2)}%
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Market Data Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h2" gutterBottom>
                Market Trend
              </Typography>
              <Box height={400}>
                <ResponsiveContainer>
                  <LineChart data={[...marketData, ...liveData]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(timestamp) => format(new Date(timestamp), 'HH:mm')}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(timestamp) => format(new Date(timestamp), 'yyyy-MM-dd HH:mm')}
                      formatter={(value: number) => [`₹${value.toLocaleString()}`, 'Price']}
                    />
                    <Line
                      type="monotone"
                      dataKey="close"
                      name="Actual"
                      stroke={theme.palette.primary.main}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted_close"
                      name="Predicted"
                      stroke={theme.palette.secondary.main}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Technical Indicators */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h2" gutterBottom>
                Technical Indicators
              </Typography>
              {liveData[0] && (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body1">
                      RSI: {liveData[0].rsi?.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body1">
                      MACD: {liveData[0].macd?.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body1">
                      Bollinger Upper: {liveData[0].bb_upper?.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body1">
                      Bollinger Lower: {liveData[0].bb_lower?.toFixed(2)}
                    </Typography>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 
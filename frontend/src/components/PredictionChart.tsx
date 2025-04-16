import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Card from 'antd/es/card';
import Statistic from 'antd/es/statistic';
import Row from 'antd/es/grid/row';
import Col from 'antd/es/grid/col';
import Spin from 'antd/es/spin';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import useInterval from '../hooks/useInterval';

interface PredictionData {
  timestamp: string;
  close: number;
  predicted_price: number;
  trend: 'up' | 'down';
  confidence: number;
  daily_volatility: number;
  avg_daily_change: number;
  rsi: number;
  volume: number;
}

const PredictionChart: React.FC = () => {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPredictionData = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/predict');
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch prediction data');
      console.error('Error fetching prediction data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchPredictionData();
  }, []);

  // Auto-refresh every 20 seconds
  useInterval(() => {
    fetchPredictionData();
  }, 20000);

  return (
    <div>
      {loading && (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin size="large" />
        </div>
      )}
      {error && (
        <div style={{ color: 'red', textAlign: 'center', padding: '20px' }}>
          {error}
        </div>
      )}
      {data && (
        <>
          <Row gutter={[16, 16]} style={{ margin: '0' }}>
            <Col xs={24} sm={12} md={6} style={{ padding: '8px' }}>
              <Card>
                <Statistic
                  title="Current Price"
                  value={data.close}
                  precision={2}
                  suffix="₹"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6} style={{ padding: '8px' }}>
              <Card>
                <Statistic
                  title="Predicted Price"
                  value={data.predicted_price}
                  precision={2}
                  valueStyle={{ color: data.trend === 'up' ? '#3f8600' : '#cf1322' }}
                  suffix="₹"
                  prefix={data.trend === 'up' ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6} style={{ padding: '8px' }}>
              <Card>
                <Statistic
                  title="Confidence"
                  value={data.confidence}
                  precision={2}
                  suffix="%"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6} style={{ padding: '8px' }}>
              <Card>
                <Statistic
                  title="Daily Volatility"
                  value={data.daily_volatility}
                  precision={2}
                  suffix="%"
                />
              </Card>
            </Col>
          </Row>

          <div style={{ marginTop: '24px', height: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[data]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="close" stroke="#8884d8" name="Current Price" />
                <Line type="monotone" dataKey="predicted_price" stroke="#82ca9d" name="Predicted Price" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
};

export default PredictionChart; 
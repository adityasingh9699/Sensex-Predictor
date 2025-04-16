import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import Card from 'antd/es/card';
import Typography from 'antd/es/typography';
import Spin from 'antd/es/spin';
import Alert from 'antd/es/alert';
import axios from 'axios';
import styled from 'styled-components';

const { Title } = Typography;

const ChartCard = styled(Card)`
  margin: 20px;
  padding: 20px;
`;

const ChartContainer = styled.div`
  margin: 30px 0;
  height: 400px;
`;

const Summary = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #f0f0f0;
`;

const Metric = styled.div`
  text-align: center;
  
  span {
    display: block;
    color: #8c8c8c;
    margin-bottom: 5px;
  }
  
  strong {
    font-size: 1.1em;
  }
`;

interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const HistoricalChart: React.FC = () => {
  const [data, setData] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/market-data');
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch historical data');
      console.error('Error fetching historical data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchHistoricalData();

    // Set up interval for automatic updates
    const intervalId = setInterval(fetchHistoricalData, 10000); // 10 seconds

    // Cleanup interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return <Spin size="large" />;
  }

  if (error) {
    return <Alert type="error" message={error} />;
  }

  if (!data.length) {
    return <Alert type="warning" message="No historical data available" />;
  }

  // Format data for the chart
  const chartData = data.map(item => ({
    ...item,
    timestamp: new Date(item.timestamp).toLocaleDateString(),
  }));

  return (
    <ChartCard>
      <Title level={3}>SENSEX Historical Data</Title>
      
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp"
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis 
              yAxisId="price"
              domain={['auto', 'auto']}
              label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft' }}
            />
            <YAxis 
              yAxisId="volume"
              orientation="right"
              domain={['auto', 'auto']}
              label={{ value: 'Volume', angle: 90, position: 'insideRight' }}
            />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="#1890ff"
              name="Closing Price"
              dot={false}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="high"
              stroke="#52c41a"
              name="High"
              dot={false}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="low"
              stroke="#f5222d"
              name="Low"
              dot={false}
            />
            <Line
              yAxisId="volume"
              type="monotone"
              dataKey="volume"
              stroke="#722ed1"
              name="Volume"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <Summary>
        <Metric>
          <span>Latest Close:</span>
          <strong>₹{data[data.length - 1].close.toFixed(2)}</strong>
        </Metric>
        <Metric>
          <span>Latest High:</span>
          <strong>₹{data[data.length - 1].high.toFixed(2)}</strong>
        </Metric>
        <Metric>
          <span>Latest Low:</span>
          <strong>₹{data[data.length - 1].low.toFixed(2)}</strong>
        </Metric>
        <Metric>
          <span>Latest Volume:</span>
          <strong>{data[data.length - 1].volume.toLocaleString()}</strong>
        </Metric>
      </Summary>
    </ChartCard>
  );
};

export default HistoricalChart; 
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface Prediction {
  trend: 'BULLISH' | 'BEARISH';
  predicted_price: number;
  confidence: number;
}

export interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi?: number;
  macd?: number;
  signal_line?: number;
  bb_upper?: number;
  bb_lower?: number;
  bb_middle?: number;
  atr?: number;
  predicted_close?: number;
}

export const getPrediction = async (daysAgo?: number): Promise<Prediction> => {
  const response = await axios.get(`${API_BASE_URL}/predict`, {
    params: { days_ago: daysAgo }
  });
  return response.data;
};

export const getMarketData = async (): Promise<MarketData[]> => {
  const response = await axios.get(`${API_BASE_URL}/historical-data`);
  return response.data;
};

export const getLiveData = async (): Promise<MarketData[]> => {
  const response = await axios.get(`${API_BASE_URL}/live-data`);
  return response.data;
}; 
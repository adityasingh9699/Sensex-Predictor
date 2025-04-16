# Sensex Predictor

A real-time stock market prediction application that uses machine learning to forecast SENSEX trends. The application provides live market data, technical indicators, and price predictions with confidence scores.

## Features

- 🚀 Real-time SENSEX price predictions
- 📊 Interactive charts with historical and predicted data
- 📈 Technical indicators (RSI, Volatility, Volume)
- 🔄 Auto-refreshing data every 20 seconds
- 📱 Responsive design for all devices

## Tech Stack

### Backend
- FastAPI (Python web framework)
- TensorFlow (Machine Learning)
- SQLAlchemy (Database ORM)
- YFinance (Market Data)
- PostgreSQL (Database)

### Frontend
- React
- TypeScript
- Ant Design
- Recharts
- Axios

## Local Development Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL
- Git

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sensex-predictor.git
cd sensex-predictor
```

2. Set up Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/sensex_db
SECRET_KEY=your_secret_key_here
```

5. Run the backend server:
```bash
uvicorn app.api.main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file:
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENV=development
```

4. Start the development server:
```bash
npm start
```

## Deployment

### Backend Deployment (Render)

1. Create a Render account at https://render.com
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `cd backend && gunicorn app.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
5. Add environment variables in Render dashboard

### Frontend Deployment (Vercel)

1. Create a Vercel account at https://vercel.com
2. Install Vercel CLI: `npm install -g vercel`
3. Deploy using Vercel CLI:
```bash
cd frontend
vercel
```
4. Add environment variables in Vercel dashboard:
```
REACT_APP_API_URL=https://your-render-backend-url
REACT_APP_ENV=production
```

## API Documentation

The API documentation is available at `/docs` or `/redoc` when running the backend server.

### Key Endpoints

- `GET /api/v1/predict` - Get real-time price predictions
- `GET /api/v1/market-data` - Get historical market data
- `GET /api/v1/live-data` - Get live market data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YFinance](https://github.com/ranaroussi/yfinance) for market data
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [Ant Design](https://ant.design/) for UI components
- [TensorFlow](https://www.tensorflow.org/) for machine learning capabilities 
# Asset Management MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Production%20Ready-brightgreen)


Production-ready MLOps pipeline for **risk prediction** and **portfolio optimization** in Asset Management. Automated, scalable, and compliant with regulatory requirements.

👉 [See all key demo screenshots here](screenshots/README.md)

## Features

### Core Capabilities

- **Automated ETL Pipeline**: Weekly data ingestion from Yahoo Finance with feature engineering (volatility, Sharpe Ratio, VaR, RSI, MACD)
- **ML Models**: Random Forest, XGBoost, Logistic Regression with AutoML for model selection
- **Risk Scoring**: Multi-level risk classification (Low/Medium/High)
- **Portfolio Optimization**: Sharpe Ratio maximization for optimal asset allocation
- **Real-time Dashboard**: Streamlit-based monitoring with interactive charts and alerts
- **REST API**: FastAPI endpoints for risk scoring, portfolio optimization, and performance prediction
- **Monitoring & Alerts**: Automated detection of volatility spikes, drawdowns, and anomalies
- **Airflow Automation**: Weekly pipeline execution with model retraining
- **Azure Deployment**: Docker, Terraform, and Azure-ready infrastructure

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry or pip
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd predict-project

# Install dependencies
poetry install
# or
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run ETL pipeline
python -m src.etl

# Launch dashboard
streamlit run src/dashboard.py

# Start API
uvicorn src.api:app --reload
```

### Docker Setup

```bash
# Build and run all services
docker-compose up -d

# Access services:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## Project Structure

```
predict-project/
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── etl.py             # ETL pipeline
│   ├── modeling.py        # ML models
│   ├── dashboard.py       # Streamlit dashboard
│   ├── api.py             # FastAPI
│   └── monitoring.py      # Monitoring system
├── airflow/               # Airflow DAGs
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/               # Saved models
├── tests/                # Unit tests
├── azure/                # Azure deployment
│   ├── deploy.sh         # Deployment script
│   └── terraform/        # IaC
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker services
└── pyproject.toml        # Dependencies
```

## Usage

### ETL Pipeline

```python
from src.etl import main

# Run complete ETL pipeline
main()
```

### Train Models

```python
from src.modeling import RiskScorer, PerformancePredictor
import pandas as pd

# Load data
df = pd.read_csv("data/processed/AAPL.csv")

# Train risk scorer
scorer = RiskScorer(model_type="xgboost")
X, y = scorer.prepare_data(df)
metrics = scorer.train(X, y)

# Train performance predictor
predictor = PerformancePredictor(model_type="xgboost")
X_perf, y_perf = predictor.prepare_data(df)
metrics_perf = predictor.train(X_perf, y_perf)
```

### API Endpoints

#### Risk Scoring

```bash
curl -X POST http://localhost:8000/api/v1/risk/score \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

#### Portfolio Optimization

```bash
curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "AMZN"], "risk_free_rate": 0.02}'
```

#### Python Client

```python
import requests

# Risk scoring
response = requests.post(
    "http://localhost:8000/api/v1/risk/score",
    json={"ticker": "AAPL"}
)
print(response.json())

# Portfolio optimization
response = requests.post(
    "http://localhost:8000/api/v1/portfolio/optimize",
    json={
        "tickers": ["AAPL", "MSFT", "AMZN"],
        "risk_free_rate": 0.02
    }
)
print(response.json())
```

## Business Metrics

The pipeline tracks:

- **Sharpe Ratio**: Risk-adjusted return
- **Value at Risk (VaR)**: Maximum potential loss at confidence level
- **Drawdown**: Maximum decline from peak
- **Volatility**: Return variability measure
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Correlations**: Inter-asset relationships

## Alert System

Automated alerts for:

- ⚠️ **Excessive volatility** (> 40%)
- 🔴 **Critical drawdown** (< -20%)
- 🟡 **Low Sharpe Ratio** (< 0.5)
- 🟡 **Overbought/Oversold** (RSI > 70 or < 30)
- ⚠️ **Significant negative return** (< -10% over 7 days)

## Architecture

```
┌─────────────────┐
│  Yahoo Finance  │
│      API        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ETL Pipeline   │  ──► data/processed/
│   (Airflow)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │  ──► MLflow Tracking
│  (RF/XGB/LR)    │
└────────┬────────┘
         │
         ├─────────┐
         │         │
         ▼         ▼
┌──────────────┐ ┌──────────────┐
│   FastAPI    │ │  Streamlit   │
│     API      │ │  Dashboard   │
└──────────────┘ └──────────────┘
         │               │
         ▼               ▼
    Monitoring & Alerts
```

## Deployment

### Azure Deployment with Terraform

```bash
cd azure/terraform
terraform init
terraform plan
terraform apply
```

### Azure Deployment with Script

```bash
cd azure
chmod +x deploy_azure.sh
./deploy_azure.sh
```

### Environment Variables

```env
# Azure Storage
AZURE_STORAGE_ACCOUNT_NAME=your-storage-account
AZURE_STORAGE_CONNECTION_STR=your-connection-string

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns

# API
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_etl.py
pytest tests/test_modeling.py
pytest tests/test_api.py

# With coverage
pytest --cov=src tests/
```

## Compliance & Governance

- **Audited logging**: All pipeline runs logged with timestamps
- **Traceability**: Model versioning with MLflow
- **Automated reports**: Weekly business reports
- **Regulatory ready**: MiFID II, ESG, EU AI Act compliant structure

## Monitoring

```python
from src.monitoring import MonitoringSystem

# Initialize monitoring
monitoring = MonitoringSystem()

# Monitor all assets
alerts = monitoring.monitor_all_assets()

# Display alerts
for ticker, ticker_alerts in alerts.items():
    for alert in ticker_alerts:
        print(f"[{alert.severity}] {alert.message}")
```

## Documentation

- [Business Case](./docs/BUSINESS_CASE.md): Business value and ROI analysis
- [User Guide](./docs/USER_GUIDE.md): Detailed usage guide
- [API Documentation](http://localhost:8000/docs): Swagger/OpenAPI docs

## Technology Stack

- **Python**: 3.11+
- **ML**: scikit-learn, XGBoost, LightGBM
- **Data**: pandas, numpy, yfinance
- **API**: FastAPI, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Orchestration**: Airflow
- **Tracking**: MLflow
- **Cloud**: Azure (Storage, App Service)
- **IaC**: Terraform
- **Containerization**: Docker, Docker Compose

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- **Data/Risk Team** - Initial development

## Acknowledgments

- Yahoo Finance for financial data access
- Open-source community for tools and libraries

---

**Note**: This project is for educational and demonstration purposes. For production use, adapt to your organization's regulatory and security requirements.

**Last Updated**: November 2025  
**Version**: 1.0.0

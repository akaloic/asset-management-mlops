# User Guide - Asset Management MLOps Pipeline

## Table of Contents

- [Installation](#installation)
- [ETL Pipeline](#etl-pipeline)
- [ML Models](#ml-models)
- [Dashboard](#dashboard)
- [REST API](#rest-api)
- [Monitoring](#monitoring)
- [Airflow Automation](#airflow-automation)
- [Azure Deployment](#azure-deployment)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.11+
- Poetry or pip
- Docker (optional)

### Local Setup

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
```

### Environment Configuration

Edit `.env` file:

```env
# Azure Storage
AZURE_STORAGE_ACCOUNT_NAME=your-storage-account
AZURE_STORAGE_CONTAINER=asset-management-data

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## ETL Pipeline

### Run Complete Pipeline

```python
from src.etl import main

# Execute full ETL pipeline
main()
```

### Programmatic Usage

```python
from src.etl import fetch_data, compute_features, save_df
from src.config import PROCESSED_DATA_DIR

# Fetch data
df = fetch_data("AAPL", start="2020-01-01")

# Compute features
df_features = compute_features(df, "AAPL")

# Save
save_df(df_features, "AAPL", PROCESSED_DATA_DIR)
```

### Generated Features

The ETL pipeline automatically calculates:

- **Returns**: Daily, log returns
- **Volatility**: 21, 63, 252-day windows (annualized)
- **Sharpe Ratio**: Multiple timeframes
- **Drawdown**: Current and maximum
- **VaR**: Value at Risk (95% confidence)
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
- **Momentum**: 5, 10, 21-day periods
- **Correlations**: Cross-asset correlation matrix

## ML Models

### Train Risk Scorer

```python
from src.modeling import RiskScorer
import pandas as pd

# Load data
df = pd.read_csv("data/processed/AAPL.csv")

# Initialize scorer
scorer = RiskScorer(model_type="xgboost")

# Prepare data
X, y = scorer.prepare_data(df, lookback_days=21)

# Train
metrics = scorer.train(X, y, test_size=0.2)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### Train Performance Predictor

```python
from src.modeling import PerformancePredictor

# Initialize predictor
predictor = PerformancePredictor(model_type="xgboost")

# Prepare data
X, y = predictor.prepare_data(df, lookforward_days=21)

# Train
metrics = predictor.train(X, y, test_size=0.2)

print(f"R²: {metrics['r2']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

### Optimize Portfolio

```python
from src.modeling import PortfolioOptimizer
import pandas as pd

# Load multiple assets
assets_data = {}
for ticker in ["AAPL", "MSFT", "AMZN"]:
    df = pd.read_csv(f"data/processed/{ticker}.csv")
    assets_data[ticker] = df['Return'].dropna()

# Align dates
returns_df = pd.DataFrame(assets_data).dropna()

# Optimize
optimizer = PortfolioOptimizer()
result = optimizer.optimize_allocation(
    returns_df,
    risk_free_rate=0.02
)

print(f"Optimal allocation: {result['allocation']}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
```

### Compare Models

```python
from src.modeling import compare_models

# Compare multiple models
results = compare_models(
    X, y,
    task_type="classification",
    models=["random_forest", "xgboost", "logistic_regression"]
)

# Display results
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")
```

## Dashboard

### Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Access at: `http://localhost:8501`

### Features

- **Real-time metrics**: Return, volatility, Sharpe Ratio, Drawdown, VaR
- **Visual alerts**: Color-coded risk indicators (red = danger, orange = warning)
- **Interactive charts**: Zoom, pan, hover for details
- **Multi-asset comparison**: Side-by-side metrics comparison

### Usage

1. Select assets from sidebar
2. View key metrics for each asset
3. Check automatic alerts for detected risks
4. Explore interactive price and drawdown charts
5. Compare metrics across assets

## REST API

### Start API Server

```bash
uvicorn src.api:app --reload
```

- **API**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`

### Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Risk Scoring

```bash
curl -X POST http://localhost:8000/api/v1/risk/score \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**Response**:
```json
{
  "ticker": "AAPL",
  "risk_score": 1,
  "risk_probability": 0.85,
  "timestamp": "2025-01-15T10:30:00",
  "metrics": {
    "volatility": 25.3,
    "sharpe_ratio": 1.2,
    "drawdown": -5.2,
    "var": -2.1
  }
}
```

#### Portfolio Optimization

```bash
curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "AMZN"],
    "risk_free_rate": 0.02
  }'
```

**Response**:
```json
{
  "allocation": {
    "AAPL": 0.35,
    "MSFT": 0.40,
    "AMZN": 0.25
  },
  "expected_return": 0.12,
  "expected_volatility": 0.18,
  "sharpe_ratio": 0.56,
  "timestamp": "2025-01-15T10:30:00"
}
```

#### Performance Prediction

```bash
curl -X POST http://localhost:8000/api/v1/performance/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### Python Client

```python
import requests

# Risk scoring
response = requests.post(
    "http://localhost:8000/api/v1/risk/score",
    json={"ticker": "AAPL"}
)
risk_data = response.json()
print(f"Risk Score: {risk_data['risk_score']}")

# Portfolio optimization
response = requests.post(
    "http://localhost:8000/api/v1/portfolio/optimize",
    json={
        "tickers": ["AAPL", "MSFT", "AMZN"],
        "risk_free_rate": 0.02
    }
)
allocation = response.json()
print(f"Allocation: {allocation['allocation']}")
```

## Monitoring

### Run Monitoring System

```python
from src.monitoring import MonitoringSystem

# Initialize monitoring
monitoring = MonitoringSystem()

# Monitor all assets
alerts = monitoring.monitor_all_assets()

# Display alerts
for ticker, ticker_alerts in alerts.items():
    if ticker_alerts:
        print(f"\n{ticker}:")
        for alert in ticker_alerts:
            print(f"  [{alert.severity}] {alert.message}")
```

### Get Recent Alerts

```python
# Get last 24 hours of alerts
recent_alerts = monitoring.get_recent_alerts(hours=24)

for alert in recent_alerts:
    print(f"{alert.asset}: {alert.message}")
```

### Alert Types

- **danger**: Immediate action required (e.g., drawdown > -20%)
- **warning**: Attention needed (e.g., high volatility)
- **info**: Useful information (e.g., oversold asset)

## Airflow Automation

### Setup Airflow

```bash
# Initialize database
airflow db init

# Create admin user
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

# Start scheduler
airflow scheduler &

# Start webserver
airflow webserver --port 8080
```

Access Airflow UI at: `http://localhost:8080`

### DAG Configuration

The `asset_management_weekly_pipeline` DAG runs automatically:

- **Schedule**: Every Monday at 2 AM
- **Tasks**:
  1. **ETL Pipeline**: Fetch and process data
  2. **Model Training**: Train ML models
  3. **Report Generation**: Generate business report
  4. **Notification**: Send completion notification

### Manual Trigger

In the Airflow UI, click "Trigger DAG" to run immediately.

## Azure Deployment

### With Docker

```bash
# Build image
docker build -t asset-management-mlops .

# Run with docker-compose
docker-compose up -d
```

### With Terraform

```bash
cd azure/terraform

# Initialize
terraform init

# Plan
terraform plan

# Deploy
terraform apply
```

### With Deployment Script

```bash
cd azure
chmod +x deploy_azure.sh
./deploy_azure.sh
```

### Azure Configuration

Set environment variables in Azure App Service:

```
AZURE_STORAGE_ACCOUNT_NAME=<storage-account>
AZURE_STORAGE_CONNECTION_STR=<connection-string>
MLFLOW_TRACKING_URI=file:./mlruns
ENVIRONMENT=production
```

## Troubleshooting

### Module Not Found

```bash
# Check installation
poetry show
# or
pip list

# Reinstall dependencies
poetry install
```

### Data Not Found

Run ETL pipeline first:
```bash
python -m src.etl
```

### Model Not Found

Train models before use:
```python
from src.modeling import RiskScorer
scorer = RiskScorer(model_type="xgboost")
# Train model first
```

### Dashboard Not Displaying

Check data exists:
```bash
ls data/processed/
```

If empty, run ETL pipeline.

### API Connection Error

Verify API is running:
```bash
curl http://localhost:8000/health
```

### Airflow DAG Not Showing

Check DAG directory:
```bash
ls airflow/dags/
```

Verify no syntax errors:
```bash
python airflow/dags/asset_management_dag.py
```

## Additional Resources

- **Technical Documentation**: See `README.md`
- **Business Case**: See `BUSINESS_CASE.md`
- **API Documentation**: http://localhost:8000/docs
- **Airflow UI**: http://localhost:8080

---

**Support**: Open an issue on the repository or contact the Data/Risk team.

**Last Updated**: November 2025  
**Version**: 1.0

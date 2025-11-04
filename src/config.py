"""
Centralized configuration for Asset Management MLOps pipeline.
Manages business parameters, Azure, paths, and models.
"""
import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Asset configuration
ASSETS: List[str] = [
    "MSFT",    # Microsoft
    "AAPL",    # Apple
    "AMZN",    # Amazon
    "SPY",     # S&P 500 ETF
    "BND",     # Total Bond ETF
    "BTC-USD"  # Bitcoin
]

# Default dates
START_DATE = "2018-01-01"
END_DATE = None  # Will use current date

# Azure configuration (via environment variables)
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "asset-management-data")
AZURE_KEY_VAULT_NAME = os.getenv("AZURE_KEY_VAULT_NAME", "")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = "Asset_Management_MLOps"

# Business configuration - Alert thresholds
ALERT_THRESHOLDS = {
    "max_volatility": 0.40,  # 40% max volatility
    "max_drawdown": -0.20,   # -20% max drawdown
    "min_sharpe_ratio": 0.5, # Minimum acceptable Sharpe
    "var_confidence": 0.05   # VaR at 95%
}

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": 42
    }
}

# Business metrics to optimize
BUSINESS_METRICS = {
    "primary": "sharpe_ratio",
    "secondary": ["var", "drawdown", "f1_score", "accuracy"]
}

# AutoML configuration
AUTOML_CONFIG = {
    "time_budget": 3600,  # 1 hour max
    "metric": "f1",
    "task": "classification",
    "eval_method": "holdout",
    "train_time_limit": 600  # 10 min max per model
}

# ETL configuration
ETL_CONFIG = {
    "update_frequency_days": 7,  # Weekly
    "lookback_window": 252,  # 1 year of trading days
    "features_to_compute": [
        "volatility",
        "drawdown",
        "sharpe_ratio",
        "var",
        "correlation",
        "rsi",
        "macd",
        "returns"
    ]
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "refresh_interval": 300,  # 5 minutes
    "max_assets_display": 20,
    "default_timeframe": "1Y"
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "title": "Asset Management ML API",
    "version": "1.0.0"
}

@dataclass
class ModelMetrics:
    """Class to store model metrics."""
    sharpe_ratio: float
    var: float
    drawdown: float
    f1_score: float
    accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "var": self.var,
            "drawdown": self.drawdown,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy
        }

# Logging configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "100 MB",
    "retention": "30 days"
}

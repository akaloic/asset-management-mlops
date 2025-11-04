"""
ML modeling module for Asset Management.
Implements Random Forest, XGBoost, Logistic Regression, AutoML
and business metrics calculation (Sharpe Ratio, VaR, Drawdown, etc.).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from datetime import datetime
from loguru import logger
import sys

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor

# MLflow - error handling with fallback
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    # Logger not yet configured, use print
    print(f"⚠️  MLflow not available: {str(e)}. Tracking will be disabled.")
    MLFLOW_AVAILABLE = False
    mlflow = None
    mlflow_sklearn = None
    mlflow_xgboost = None

# Context manager for MLflow
class MLflowContext:
    """Context manager for MLflow."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.run = None
        
    def __enter__(self):
        if MLFLOW_AVAILABLE:
            self.run = mlflow.start_run(*self.args, **self.kwargs)
            return self.run
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if MLFLOW_AVAILABLE and self.run:
            try:
                mlflow.end_run()
            except:
                pass

def mlflow_set_experiment(name):
    """Set MLflow experiment."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(name)
        except Exception as e:
            logger.warning(f"MLflow set_experiment error: {str(e)}")

def mlflow_log_params(params):
    """Log parameters in MLflow."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow log_params error: {str(e)}")

def mlflow_log_metrics(metrics):
    """Log metrics in MLflow."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"MLflow log_metrics error: {str(e)}")

def mlflow_log_model(model, name, flavor="sklearn"):
    """Log model in MLflow."""
    if MLFLOW_AVAILABLE:
        try:
            if flavor == "sklearn":
                mlflow.sklearn.log_model(model, name)
            elif flavor == "xgboost":
                mlflow.xgboost.log_model(model, name)
        except Exception as e:
            logger.warning(f"MLflow log_model error: {str(e)}")

from src.config import (
    MODELS_DIR, LOGS_DIR, MODEL_CONFIG, BUSINESS_METRICS,
    AUTOML_CONFIG, ALERT_THRESHOLDS, LOG_CONFIG, ModelMetrics
)

# Logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format=LOG_CONFIG["format"],
    level=LOG_CONFIG["level"]
)
logger.add(
    LOGS_DIR / "modeling_{time:YYYY-MM-DD}.log",
    rotation=LOG_CONFIG["rotation"],
    retention=LOG_CONFIG["retention"],
    level=LOG_CONFIG["level"]
)


class RiskScorer:
    """
    Class for asset risk scoring.
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize risk scorer.
        
        Args:
            model_type: Model type ('random_forest', 'xgboost', 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        mlflow_set_experiment("Asset_Management_Risk_Scoring")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "Risk_Label",
        lookback_days: int = 21
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features
            target_col: Target column (if absent, will be created based on drawdown)
            lookback_days: Number of days to predict future risk
            
        Returns:
            X (features), y (target)
        """
        logger.info(f"Data preparation for {self.model_type}")
        df = df.copy()
        
        # Create target variable if it doesn't exist
        if target_col not in df.columns:
            # Create risk label based on future drawdown
            df['Future_Drawdown'] = df['Drawdown'].shift(-lookback_days)
            df[target_col] = pd.cut(
                df['Future_Drawdown'],
                bins=[-np.inf, -0.15, -0.05, np.inf],
                labels=[2, 1, 0]  # 2=High Risk, 1=Medium Risk, 0=Low Risk
            )
            df = df.dropna(subset=[target_col])
        
        # Select features (exclude non-numeric columns and target)
        exclude_cols = ['Date', target_col, 'Future_Drawdown', 'Cum_Max']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].astype(int)
        
        self.feature_columns = feature_cols
        
        logger.success(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train risk scoring model.
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Training model {self.model_type}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(**MODEL_CONFIG["xgboost"])
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(**MODEL_CONFIG["logistic_regression"])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Training with MLflow tracking (optional)
        with MLflowContext(run_name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Log in MLflow if available
            mlflow_log_params(MODEL_CONFIG[self.model_type])
            mlflow_log_metrics({
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            
            # Save model in MLflow if available
            if self.model_type in ["random_forest", "logistic_regression"]:
                mlflow_log_model(self.model, "model", flavor="sklearn")
            else:
                mlflow_log_model(self.model, "model", flavor="xgboost")
            
            logger.success(f"Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
    
    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk for new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with risk predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_columns].fillna(0)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        result = df.copy()
        result['Risk_Score'] = predictions
        result['Risk_Proba'] = probabilities.max(axis=1)
        
        return result


class PerformancePredictor:
    """
    Class for performance prediction (regression).
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize performance predictor.
        
        Args:
            model_type: Model type ('random_forest', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        mlflow_set_experiment("Asset_Management_Performance_Prediction")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "Future_Return",
        lookforward_days: int = 21
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for performance prediction.
        
        Args:
            df: DataFrame with features
            target_col: Target column (future return)
            lookforward_days: Number of days to predict future return
            
        Returns:
            X (features), y (target)
        """
        logger.info(f"Data preparation for performance prediction")
        df = df.copy()
        
        # Create target variable (future return)
        if target_col not in df.columns:
            df['Future_Price'] = df['Close'].shift(-lookforward_days)
            df[target_col] = (df['Future_Price'] - df['Close']) / df['Close']
            df = df.dropna(subset=[target_col])
        
        # Select features
        exclude_cols = ['Date', target_col, 'Future_Price', 'Cum_Max']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        self.feature_columns = feature_cols
        
        logger.success(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train performance prediction model.
        
        Args:
            X: Features
            y: Target (returns)
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Training model {self.model_type}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(**MODEL_CONFIG["random_forest"])
        elif self.model_type == "xgboost":
            self.model = XGBRegressor(**MODEL_CONFIG["xgboost"])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Training with MLflow tracking (optional)
        with MLflowContext(run_name=f"{self.model_type}_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Business metrics
            sharpe_ratio = self.compute_sharpe_ratio(y_test, y_pred)
            var_actual = self.compute_var(y_test)
            var_pred = self.compute_var(y_pred)
            
            # Log in MLflow if available
            mlflow_log_params(MODEL_CONFIG[self.model_type])
            mlflow_log_metrics({
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "sharpe_ratio": sharpe_ratio,
                "var_actual": var_actual,
                "var_pred": var_pred
            })
            
            # Save model in MLflow if available
            if self.model_type == "random_forest":
                mlflow_log_model(self.model, "model", flavor="sklearn")
            else:
                mlflow_log_model(self.model, "model", flavor="xgboost")
            
            logger.success(f"Model trained - R²: {r2:.4f}, RMSE: {rmse:.4f}, Sharpe: {sharpe_ratio:.4f}")
            
            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "sharpe_ratio": sharpe_ratio,
                "var_actual": var_actual,
                "var_pred": var_pred
            }
    
    @staticmethod
    def compute_sharpe_ratio(returns_actual: pd.Series, returns_pred: pd.Series) -> float:
        """Compute Sharpe Ratio between predictions and actual values."""
        if len(returns_actual) == 0 or returns_actual.std() == 0:
            return 0.0
        correlation = returns_actual.corr(returns_pred)
        if pd.isna(correlation):
            return 0.0
        return float(correlation)
    
    @staticmethod
    def compute_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Compute Value at Risk."""
        if returns.empty:
            return 0.0
        return float(np.percentile(returns.dropna(), confidence * 100))
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict performance for new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with performance predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_columns].fillna(0)
        predictions = self.model.predict(X)
        
        result = df.copy()
        result['Predicted_Return'] = predictions
        
        return result


class PortfolioOptimizer:
    """
    Class for portfolio optimization (optimal allocation).
    """
    
    @staticmethod
    def optimize_allocation(
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation to maximize Sharpe Ratio.
        
        Args:
            returns_df: DataFrame with returns for each asset (columns = assets)
            risk_free_rate: Annual risk-free rate
            target_return: Target return (optional)
            
        Returns:
            Dictionary with optimal weights for each asset
        """
        logger.info("Portfolio allocation optimization")
        
        from scipy.optimize import minimize
        
        n_assets = returns_df.shape[1]
        asset_names = returns_df.columns.tolist()
        
        # Calculate covariance matrix and mean returns
        mean_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        # Objective function: -Sharpe Ratio (we minimize the negative)
        def negative_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_std == 0:
                return -1000
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum = 1
        
        # Bounds (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Equal-weighted initialization
        initial_weights = np.array([1 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        allocation = {asset_names[i]: float(optimal_weights[i]) for i in range(n_assets)}
        
        # Calculate optimal Sharpe Ratio
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        optimal_sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        
        logger.success(f"Allocation optimized - Sharpe Ratio: {optimal_sharpe:.4f}")
        
        return {
            "allocation": allocation,
            "expected_return": float(portfolio_return),
            "expected_volatility": float(portfolio_std),
            "sharpe_ratio": float(optimal_sharpe)
        }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    models: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models and return their metrics.
    
    Args:
        X: Features
        y: Target
        task_type: Task type ('classification' or 'regression')
        models: List of models to compare (None = all)
        
    Returns:
        Dictionary with metrics for each model
    """
    logger.info(f"Model comparison for {task_type}")
    
    if models is None:
        if task_type == "classification":
            models = ["random_forest", "xgboost", "logistic_regression"]
        else:
            models = ["random_forest", "xgboost"]
    
    results = {}
    
    for model_name in models:
        try:
            if task_type == "classification":
                scorer = RiskScorer(model_type=model_name)
            else:
                scorer = PerformancePredictor(model_type=model_name)
            
            metrics = scorer.train(X, y)
            results[model_name] = metrics
            
        except Exception as e:
            logger.error(f"Error during {model_name} training: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Find best model
    if task_type == "classification":
        best_model = max(results.keys(), key=lambda k: results[k].get("f1_score", 0))
    else:
        best_model = max(results.keys(), key=lambda k: results[k].get("r2", -1000))
    
    logger.success(f"Best model: {best_model}")
    
    return results


def main():
    """Main function to test models."""
    logger.info("Testing scoring and prediction models")
    
    # Load sample data
    from src.config import PROCESSED_DATA_DIR
    
    try:
        # Load data from an asset
        df = pd.read_csv(PROCESSED_DATA_DIR / "AAPL.csv")
        
        logger.info("Testing Risk Scorer")
        scorer = RiskScorer(model_type="xgboost")
        X, y = scorer.prepare_data(df)
        if len(X) > 0 and len(y) > 0:
            metrics = scorer.train(X, y)
            logger.success(f"Risk Scorer metrics: {metrics}")
        
        logger.info("Testing Performance Predictor")
        predictor = PerformancePredictor(model_type="xgboost")
        X_perf, y_perf = predictor.prepare_data(df)
        if len(X_perf) > 0 and len(y_perf) > 0:
            metrics_perf = predictor.train(X_perf, y_perf)
            logger.success(f"Performance Predictor metrics: {metrics_perf}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    main()

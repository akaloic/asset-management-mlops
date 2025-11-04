"""
Monitoring and alerting module for Asset Management.
Monitors key indicator variations and generates alerts.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import sys
import json

from src.config import (
    PROCESSED_DATA_DIR, LOGS_DIR, ALERT_THRESHOLDS,
    ASSETS, LOG_CONFIG
)

# Logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format=LOG_CONFIG["format"],
    level=LOG_CONFIG["level"]
)
logger.add(
    LOGS_DIR / "monitoring_{time:YYYY-MM-DD}.log",
    rotation=LOG_CONFIG["rotation"],
    retention=LOG_CONFIG["retention"],
    level=LOG_CONFIG["level"]
)


class Alert:
    """Class to represent an alert."""
    
    def __init__(self, asset: str, alert_type: str, message: str, severity: str = "warning"):
        """
        Initialize an alert.
        
        Args:
            asset: Asset symbol
            alert_type: Alert type (volatility, drawdown, sharpe, etc.)
            message: Alert message
            severity: Severity level (info, warning, danger)
        """
        self.asset = asset
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            "asset": self.asset,
            "alert_type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.asset} - {self.message}"


class MonitoringSystem:
    """Monitoring system for Asset Management."""
    
    def __init__(self):
        """Initialize monitoring system."""
        self.alerts_log_file = LOGS_DIR / "alerts.jsonl"
        self.metrics_log_file = LOGS_DIR / "metrics.jsonl"
    
    def load_data(self, ticker: str) -> pd.DataFrame:
        """Load asset data."""
        file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
        if file_path.exists():
            return pd.read_csv(file_path, parse_dates=['Date'])
        return pd.DataFrame()
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for an asset."""
        if df.empty:
            return {}
        
        metrics = {}
        latest = df.iloc[-1]
        
        # Volatility
        if 'Volatility_21d' in df.columns:
            metrics['volatility'] = float(latest['Volatility_21d'] * 100) if pd.notna(latest['Volatility_21d']) else 0.0
            # Average volatility over 30 days
            metrics['volatility_30d_avg'] = float(df['Volatility_21d'].tail(30).mean() * 100) if len(df) >= 30 else metrics['volatility']
        
        # Drawdown
        if 'Drawdown' in df.columns:
            metrics['current_drawdown'] = float(latest['Drawdown'] * 100) if pd.notna(latest['Drawdown']) else 0.0
            metrics['max_drawdown'] = float(df['Drawdown'].min() * 100) if 'Drawdown' in df.columns else 0.0
        
        # Sharpe Ratio
        if 'Sharpe_21d' in df.columns:
            metrics['sharpe_ratio'] = float(latest['Sharpe_21d']) if pd.notna(latest['Sharpe_21d']) else 0.0
            metrics['sharpe_30d_avg'] = float(df['Sharpe_21d'].tail(30).mean()) if len(df) >= 30 else metrics['sharpe_ratio']
        
        # VaR
        if 'VaR_21d' in df.columns:
            metrics['var'] = float(latest['VaR_21d'] * 100) if pd.notna(latest['VaR_21d']) else 0.0
        
        # RSI
        if 'RSI_14' in df.columns:
            metrics['rsi'] = float(latest['RSI_14']) if pd.notna(latest['RSI_14']) else 50.0
        
        # Return
        if 'Return' in df.columns:
            returns = df['Return'].dropna()
            metrics['return_1d'] = float(returns.iloc[-1] * 100) if len(returns) > 0 else 0.0
            metrics['return_7d'] = float(returns.tail(7).sum() * 100) if len(returns) >= 7 else 0.0
            metrics['return_30d'] = float(returns.tail(30).sum() * 100) if len(returns) >= 30 else 0.0
        
        return metrics
    
    def check_alerts(self, ticker: str, metrics: Dict[str, float]) -> List[Alert]:
        """Check alerts for an asset."""
        alerts = []
        
        # Excessive volatility alert
        if 'volatility' in metrics and metrics['volatility'] > ALERT_THRESHOLDS['max_volatility'] * 100:
            alerts.append(Alert(
                asset=ticker,
                alert_type="volatility",
                message=f"Excessive volatility: {metrics['volatility']:.2f}% (threshold: {ALERT_THRESHOLDS['max_volatility']*100}%)",
                severity="danger"
            ))
        elif 'volatility' in metrics and 'volatility_30d_avg' in metrics:
            # Check if volatility has increased significantly
            if metrics['volatility'] > metrics['volatility_30d_avg'] * 1.5:
                alerts.append(Alert(
                    asset=ticker,
                    alert_type="volatility_spike",
                    message=f"Volatility spike: {metrics['volatility']:.2f}% vs 30d avg: {metrics['volatility_30d_avg']:.2f}%",
                    severity="warning"
                ))
        
        # Critical drawdown alert
        if 'current_drawdown' in metrics and metrics['current_drawdown'] < ALERT_THRESHOLDS['max_drawdown'] * 100:
            alerts.append(Alert(
                asset=ticker,
                alert_type="drawdown",
                message=f"Critical drawdown: {metrics['current_drawdown']:.2f}% (threshold: {ALERT_THRESHOLDS['max_drawdown']*100}%)",
                severity="danger"
            ))
        elif 'current_drawdown' in metrics and metrics['current_drawdown'] < ALERT_THRESHOLDS['max_drawdown'] * 100 * 0.8:
            alerts.append(Alert(
                asset=ticker,
                alert_type="drawdown",
                message=f"High drawdown: {metrics['current_drawdown']:.2f}%",
                severity="warning"
            ))
        
        # Low Sharpe Ratio alert
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] < ALERT_THRESHOLDS['min_sharpe_ratio']:
            alerts.append(Alert(
                asset=ticker,
                alert_type="sharpe_ratio",
                message=f"Low Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (threshold: {ALERT_THRESHOLDS['min_sharpe_ratio']})",
                severity="warning"
            ))
        
        # RSI alert (overbought/oversold)
        if 'rsi' in metrics:
            if metrics['rsi'] > 70:
                alerts.append(Alert(
                    asset=ticker,
                    alert_type="rsi",
                    message=f"Overbought asset (RSI: {metrics['rsi']:.1f})",
                    severity="warning"
                ))
            elif metrics['rsi'] < 30:
                alerts.append(Alert(
                    asset=ticker,
                    alert_type="rsi",
                    message=f"Oversold asset (RSI: {metrics['rsi']:.1f})",
                    severity="info"
                ))
        
        # Significant negative return alert
        if 'return_7d' in metrics and metrics['return_7d'] < -10:
            alerts.append(Alert(
                asset=ticker,
                alert_type="return",
                message=f"Very negative weekly return: {metrics['return_7d']:.2f}%",
                severity="warning"
            ))
        
        return alerts
    
    def detect_anomalies(self, df: pd.DataFrame, ticker: str) -> List[Alert]:
        """Detect anomalies in data."""
        alerts = []
        
        if df.empty:
            return alerts
        
        # Detect significant missing values
        missing_pct = df.isna().sum() / len(df) * 100
        if any(missing_pct > 20):
            cols_with_missing = missing_pct[missing_pct > 20].index.tolist()
            alerts.append(Alert(
                asset=ticker,
                alert_type="data_quality",
                message=f"Significant missing data: {', '.join(cols_with_missing)}",
                severity="warning"
            ))
        
        # Detect outliers in returns
        if 'Return' in df.columns:
            returns = df['Return'].dropna()
            if len(returns) > 0:
                mean_return = returns.mean()
                std_return = returns.std()
                outliers = returns[abs(returns - mean_return) > 3 * std_return]
                if len(outliers) > len(returns) * 0.05:  # More than 5% outliers
                    alerts.append(Alert(
                        asset=ticker,
                        alert_type="outliers",
                        message=f"Significant number of outliers detected in returns",
                        severity="info"
                    ))
        
        return alerts
    
    def log_alert(self, alert: Alert) -> None:
        """Log an alert."""
        with open(self.alerts_log_file, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')
    
    def log_metrics(self, ticker: str, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        log_entry = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        with open(self.metrics_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def monitor_all_assets(self) -> Dict[str, List[Alert]]:
        """Monitor all assets and generate alerts."""
        logger.info("=" * 60)
        logger.info("STARTING MONITORING")
        logger.info("=" * 60)
        
        all_alerts = {}
        
        for ticker in ASSETS:
            logger.info(f"Monitoring {ticker}")
            
            df = self.load_data(ticker)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            
            # Calculate metrics
            metrics = self.compute_metrics(df)
            self.log_metrics(ticker, metrics)
            
            # Check alerts
            alerts = self.check_alerts(ticker, metrics)
            
            # Detect anomalies
            anomaly_alerts = self.detect_anomalies(df, ticker)
            alerts.extend(anomaly_alerts)
            
            # Log alerts
            for alert in alerts:
                self.log_alert(alert)
                if alert.severity == "danger":
                    logger.error(str(alert))
                elif alert.severity == "warning":
                    logger.warning(str(alert))
                else:
                    logger.info(str(alert))
            
            all_alerts[ticker] = alerts
        
        logger.info("=" * 60)
        logger.success("MONITORING COMPLETED")
        logger.info("=" * 60)
        
        # Summary
        total_alerts = sum(len(alerts) for alerts in all_alerts.values())
        danger_alerts = sum(
            len([a for a in alerts if a.severity == "danger"])
            for alerts in all_alerts.values()
        )
        warning_alerts = sum(
            len([a for a in alerts if a.severity == "warning"])
            for alerts in all_alerts.values()
        )
        
        logger.info(f"Total alerts: {total_alerts} (Danger: {danger_alerts}, Warning: {warning_alerts})")
        
        return all_alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Retrieve recent alerts."""
        if not self.alerts_log_file.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = []
        
        try:
            with open(self.alerts_log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        alert_dict = json.loads(line)
                        alert_time = datetime.fromisoformat(alert_dict['timestamp'])
                        if alert_time >= cutoff_time:
                            alerts.append(Alert(
                                asset=alert_dict['asset'],
                                alert_type=alert_dict['alert_type'],
                                message=alert_dict['message'],
                                severity=alert_dict['severity']
                            ))
        except Exception as e:
            logger.error(f"Error reading alerts: {str(e)}")
        
        return alerts


def main():
    """Main function to test monitoring."""
    monitoring = MonitoringSystem()
    alerts = monitoring.monitor_all_assets()
    
    # Display summary
    print("\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)
    for ticker, ticker_alerts in alerts.items():
        if ticker_alerts:
            print(f"\n{ticker}: {len(ticker_alerts)} alert(s)")
            for alert in ticker_alerts:
                print(f"  - {alert}")


if __name__ == "__main__":
    main()

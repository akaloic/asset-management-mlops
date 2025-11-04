"""
Streamlit Dashboard for Asset Management business monitoring.
Displays risk scores, performance, alerts and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import sys

# Page configuration
st.set_page_config(
    page_title="Asset Management MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, ASSETS,
    ALERT_THRESHOLDS, DASHBOARD_CONFIG
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-danger {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)


def load_data(ticker: str) -> pd.DataFrame:
    """Load asset data."""
    file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
    if file_path.exists():
        return pd.read_csv(file_path, parse_dates=['Date'])
    return pd.DataFrame()


def compute_business_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate business metrics for an asset."""
    if df.empty:
        return {}
    
    metrics = {}
    
    # Returns
    if 'Return' in df.columns:
        returns = df['Return'].dropna()
        metrics['total_return'] = float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100)
        metrics['annualized_return'] = float(returns.mean() * 252 * 100)
        metrics['volatility'] = float(returns.std() * np.sqrt(252) * 100)
    
    # Sharpe Ratio
    if 'Sharpe_21d' in df.columns:
        metrics['sharpe_ratio'] = float(df['Sharpe_21d'].iloc[-1]) if not df['Sharpe_21d'].isna().all() else 0.0
    
    # Drawdown
    if 'Drawdown' in df.columns:
        metrics['current_drawdown'] = float(df['Drawdown'].iloc[-1] * 100)
        metrics['max_drawdown'] = float(df['Drawdown'].min() * 100)
    
    # VaR
    if 'Return' in df.columns:
        returns = df['Return'].dropna()
        if len(returns) > 0:
            metrics['var_95'] = float(np.percentile(returns, 5) * 100)
    
    # RSI
    if 'RSI_14' in df.columns:
        metrics['rsi'] = float(df['RSI_14'].iloc[-1]) if not df['RSI_14'].isna().all() else 50.0
    
    return metrics


def check_alerts(metrics: Dict[str, float]) -> List[Dict[str, str]]:
    """Check alerts based on thresholds."""
    alerts = []
    
    if 'volatility' in metrics:
        if metrics['volatility'] > ALERT_THRESHOLDS['max_volatility'] * 100:
            alerts.append({
                'type': 'danger',
                'message': f"⚠️ Excessive volatility: {metrics['volatility']:.2f}% (> {ALERT_THRESHOLDS['max_volatility']*100}%)"
            })
    
    if 'current_drawdown' in metrics:
        if metrics['current_drawdown'] < ALERT_THRESHOLDS['max_drawdown'] * 100:
            alerts.append({
                'type': 'danger',
                'message': f"🔴 Critical drawdown: {metrics['current_drawdown']:.2f}% (< {ALERT_THRESHOLDS['max_drawdown']*100}%)"
            })
        elif metrics['current_drawdown'] < ALERT_THRESHOLDS['max_drawdown'] * 100 * 0.8:
            alerts.append({
                'type': 'warning',
                'message': f"🟡 High drawdown: {metrics['current_drawdown']:.2f}%"
            })
    
    if 'sharpe_ratio' in metrics:
        if metrics['sharpe_ratio'] < ALERT_THRESHOLDS['min_sharpe_ratio']:
            alerts.append({
                'type': 'warning',
                'message': f"🟡 Low Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (< {ALERT_THRESHOLDS['min_sharpe_ratio']})"
            })
    
    if 'rsi' in metrics:
        if metrics['rsi'] > 70:
            alerts.append({
                'type': 'warning',
                'message': f"🟡 Overbought asset (RSI: {metrics['rsi']:.1f})"
            })
        elif metrics['rsi'] < 30:
            alerts.append({
                'type': 'info',
                'message': f"ℹ️ Oversold asset (RSI: {metrics['rsi']:.1f})"
            })
    
    return alerts


def plot_price_chart(df: pd.DataFrame, ticker: str):
    """Plot price chart with indicators."""
    if df.empty or 'Close' not in df.columns:
        return
    
    fig = go.Figure()
    
    # Closing price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Moving Averages
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{ticker} - Price Evolution",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_comparison(assets_data: Dict[str, Dict[str, float]]):
    """Compare metrics across assets."""
    if not assets_data:
        return
    
    # Prepare data
    metrics_df = pd.DataFrame(assets_data).T
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sharpe_ratio' in metrics_df.columns:
            fig = px.bar(
                metrics_df.reset_index(),
                x='index',
                y='sharpe_ratio',
                title='Sharpe Ratio by Asset',
                labels={'index': 'Asset', 'sharpe_ratio': 'Sharpe Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'volatility' in metrics_df.columns:
            fig = px.bar(
                metrics_df.reset_index(),
                x='index',
                y='volatility',
                title='Volatility by Asset',
                labels={'index': 'Asset', 'volatility': 'Volatility (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)


def plot_drawdown_chart(df: pd.DataFrame, ticker: str):
    """Plot drawdown chart."""
    if df.empty or 'Drawdown' not in df.columns:
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Drawdown'] * 100,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='red', width=2),
        fillcolor='rgba(255,0,0,0.3)'
    ))
    
    fig.update_layout(
        title=f"{ticker} - Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">📊 Asset Management MLOps Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Asset selection
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        ASSETS,
        default=ASSETS[:3] if len(ASSETS) >= 3 else ASSETS
    )
    
    # Display options
    show_alerts = st.sidebar.checkbox("Show Alerts", value=True)
    show_comparison = st.sidebar.checkbox("Compare Assets", value=True)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    if not selected_assets:
        st.warning("Please select at least one asset.")
        return
    
    # Load and display data for each asset
    assets_metrics = {}
    
    for ticker in selected_assets:
        st.markdown(f"## {ticker}")
        
        df = load_data(ticker)
        
        if df.empty:
            st.error(f"No data available for {ticker}")
            continue
        
        # Calculate metrics
        metrics = compute_business_metrics(df)
        assets_metrics[ticker] = metrics
        
        # Display alerts
        if show_alerts:
            alerts = check_alerts(metrics)
            if alerts:
                for alert in alerts:
                    if alert['type'] == 'danger':
                        st.markdown(f'<div class="alert-danger">{alert["message"]}</div>', unsafe_allow_html=True)
                    elif alert['type'] == 'warning':
                        st.markdown(f'<div class="alert-warning">{alert["message"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info(alert['message'])
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2f}%",
                delta=f"{metrics.get('annualized_return', 0):.2f}% annual"
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{metrics.get('volatility', 0):.2f}%",
                delta="Annual"
            )
        
        with col3:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="21 days"
            )
        
        with col4:
            st.metric(
                "Current Drawdown",
                f"{metrics.get('current_drawdown', 0):.2f}%",
                delta=f"Max: {metrics.get('max_drawdown', 0):.2f}%"
            )
        
        with col5:
            st.metric(
                "VaR 95%",
                f"{metrics.get('var_95', 0):.2f}%",
                delta="Daily"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            plot_price_chart(df, ticker)
        
        with col2:
            plot_drawdown_chart(df, ticker)
        
        st.markdown("---")
    
    # Asset comparison
    if show_comparison and len(assets_metrics) > 1:
        st.markdown("## Asset Comparison")
        plot_metrics_comparison(assets_metrics)
        
        # Summary table
        st.markdown("### Summary Table")
        summary_df = pd.DataFrame(assets_metrics).T
        st.dataframe(summary_df.style.highlight_max(axis=0, subset=[col for col in summary_df.columns if col != 'index']))


if __name__ == "__main__":
    main()

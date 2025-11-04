"""
API FastAPI pour scoring live et optimisation de portefeuille.
Expose les modèles ML et les fonctions d'optimisation via une API REST.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from loguru import logger

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, ASSETS,
    ALERT_THRESHOLDS, API_CONFIG, LOG_CONFIG, LOGS_DIR
)

# Configuration du logger
logger.remove()
logger.add(
    sys.stdout,
    format=LOG_CONFIG["format"],
    level=LOG_CONFIG["level"]
)
logger.add(
    LOGS_DIR / "api_{time:YYYY-MM-DD}.log",
    rotation=LOG_CONFIG["rotation"],
    retention=LOG_CONFIG["retention"],
    level=LOG_CONFIG["level"]
)

# Initialiser FastAPI
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description="API MLOps pour Asset Management - Scoring de risque et optimisation de portefeuille"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modèles Pydantic pour les requêtes/réponses
class RiskScoreRequest(BaseModel):
    """Requête pour le scoring de risque."""
    ticker: str = Field(..., description="Symbole du ticker")
    features: Optional[Dict[str, float]] = Field(None, description="Features optionnelles")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "features": None
            }
        }


class RiskScoreResponse(BaseModel):
    """Réponse du scoring de risque."""
    ticker: str
    risk_score: int = Field(..., description="0=Low Risk, 1=Medium Risk, 2=High Risk")
    risk_probability: float = Field(..., description="Probabilité du score de risque")
    timestamp: str
    metrics: Dict[str, float] = Field(..., description="Métriques métier")


class PortfolioOptimizationRequest(BaseModel):
    """Requête pour l'optimisation de portefeuille."""
    tickers: List[str] = Field(..., description="Liste des tickers du portefeuille")
    risk_free_rate: float = Field(0.02, description="Taux sans risque annuel")
    target_return: Optional[float] = Field(None, description="Return cible (optionnel)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["AAPL", "MSFT", "AMZN"],
                "risk_free_rate": 0.02,
                "target_return": None
            }
        }


class PortfolioOptimizationResponse(BaseModel):
    """Réponse de l'optimisation de portefeuille."""
    allocation: Dict[str, float] = Field(..., description="Allocation optimale")
    expected_return: float = Field(..., description="Return attendu annualisé")
    expected_volatility: float = Field(..., description="Volatilité attendue annualisée")
    sharpe_ratio: float = Field(..., description="Sharpe Ratio optimal")
    timestamp: str


class PerformancePredictionRequest(BaseModel):
    """Requête pour la prédiction de performance."""
    ticker: str = Field(..., description="Symbole du ticker")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL"
            }
        }


class PerformancePredictionResponse(BaseModel):
    """Réponse de la prédiction de performance."""
    ticker: str
    predicted_return: float = Field(..., description="Return prédit")
    confidence_interval: Dict[str, float] = Field(..., description="Intervalle de confiance")
    timestamp: str


# Charger les modèles (lazy loading)
risk_scorers = {}
performance_predictors = {}


def load_models():
    """Charge les modèles ML (à appeler au démarrage)."""
    logger.info("Chargement des modèles ML")
    # Les modèles seront chargés à la demande
    # Dans une vraie production, charger depuis MLflow ou Azure ML
    pass


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API."""
    logger.info("Démarrage de l'API Asset Management MLOps")
    load_models()


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "message": "API Asset Management MLOps",
        "version": API_CONFIG["version"],
        "endpoints": {
            "risk_scoring": "/api/v1/risk/score",
            "portfolio_optimization": "/api/v1/portfolio/optimize",
            "performance_prediction": "/api/v1/performance/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(risk_scorers) > 0
    }


@app.post("/api/v1/risk/score", response_model=RiskScoreResponse)
async def score_risk(request: RiskScoreRequest):
    """
    Score le risque d'un actif.
    
    Args:
        request: Requête avec le ticker et features optionnelles
        
    Returns:
        Score de risque avec métriques métier
    """
    try:
        logger.info(f"Scoring de risque pour {request.ticker}")
        
        # Charger les données de l'actif
        file_path = PROCESSED_DATA_DIR / f"{request.ticker}.csv"
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Données non trouvées pour {request.ticker}"
            )
        
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Données vides pour {request.ticker}"
            )
        
        # Utiliser les dernières données
        latest_data = df.iloc[-1:].copy()
        
        # Si un modèle est disponible, l'utiliser
        # Sinon, calculer un score basé sur les métriques
        if request.ticker in risk_scorers:
            scorer = risk_scorers[request.ticker]
            result_df = scorer.predict_risk(latest_data)
            risk_score = int(result_df['Risk_Score'].iloc[0])
            risk_probability = float(result_df['Risk_Proba'].iloc[0])
        else:
            # Scoring basique basé sur le drawdown
            drawdown = latest_data['Drawdown'].iloc[0] if 'Drawdown' in latest_data.columns else 0
            if drawdown < -0.15:
                risk_score = 2  # High Risk
            elif drawdown < -0.05:
                risk_score = 1  # Medium Risk
            else:
                risk_score = 0  # Low Risk
            risk_probability = abs(drawdown)
        
        # Calculer les métriques métier
        metrics = {}
        if 'Volatility_21d' in latest_data.columns:
            metrics['volatility'] = float(latest_data['Volatility_21d'].iloc[0] * 100)
        if 'Sharpe_21d' in latest_data.columns:
            metrics['sharpe_ratio'] = float(latest_data['Sharpe_21d'].iloc[0])
        if 'Drawdown' in latest_data.columns:
            metrics['drawdown'] = float(latest_data['Drawdown'].iloc[0] * 100)
        if 'VaR_21d' in latest_data.columns:
            metrics['var'] = float(latest_data['VaR_21d'].iloc[0] * 100)
        
        return RiskScoreResponse(
            ticker=request.ticker,
            risk_score=risk_score,
            risk_probability=risk_probability,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du scoring de risque: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """
    Optimise l'allocation d'un portefeuille.
    
    Args:
        request: Requête avec les tickers et paramètres
        
    Returns:
        Allocation optimale avec métriques
    """
    try:
        logger.info(f"Optimisation du portefeuille: {request.tickers}")
        
        # Charger les données de tous les actifs
        returns_data = {}
        
        for ticker in request.tickers:
            file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
            if not file_path.exists():
                logger.warning(f"Données non trouvées pour {ticker}, ignoré")
                continue
            
            df = pd.read_csv(file_path, parse_dates=['Date'])
            if df.empty or 'Return' not in df.columns:
                logger.warning(f"Données invalides pour {ticker}, ignoré")
                continue
            
            returns_data[ticker] = df['Return'].dropna()
        
        if len(returns_data) < 2:
            raise HTTPException(
                status_code=400,
                detail="Au moins 2 actifs avec données valides sont requis"
            )
        
        # Aligner les dates
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Aucune donnée commune trouvée entre les actifs"
            )
        
        # Optimiser l'allocation
        from src.modeling import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_allocation(
            returns_df,
            risk_free_rate=request.risk_free_rate,
            target_return=request.target_return
        )
        
        return PortfolioOptimizationResponse(
            allocation=result["allocation"],
            expected_return=result["expected_return"],
            expected_volatility=result["expected_volatility"],
            sharpe_ratio=result["sharpe_ratio"],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/performance/predict", response_model=PerformancePredictionResponse)
async def predict_performance(request: PerformancePredictionRequest):
    """
    Prédit la performance future d'un actif.
    
    Args:
        request: Requête avec le ticker
        
    Returns:
        Prédiction de performance avec intervalle de confiance
    """
    try:
        logger.info(f"Prédiction de performance pour {request.ticker}")
        
        # Charger les données de l'actif
        file_path = PROCESSED_DATA_DIR / f"{request.ticker}.csv"
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Données non trouvées pour {request.ticker}"
            )
        
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Données vides pour {request.ticker}"
            )
        
        # Utiliser les dernières données
        latest_data = df.iloc[-1:].copy()
        
        # Si un modèle est disponible, l'utiliser
        # Sinon, estimer basé sur le return historique
        if request.ticker in performance_predictors:
            predictor = performance_predictors[request.ticker]
            result_df = predictor.predict(latest_data)
            predicted_return = float(result_df['Predicted_Return'].iloc[0])
        else:
            # Estimation basique basée sur le return historique moyen
            if 'Return' in df.columns:
                predicted_return = float(df['Return'].mean() * 21)  # 21 jours
            else:
                predicted_return = 0.0
        
        # Calculer l'intervalle de confiance (basique)
        if 'Return' in df.columns:
            returns_std = df['Return'].std()
            confidence_interval = {
                "lower": float(predicted_return - 1.96 * returns_std * np.sqrt(21)),
                "upper": float(predicted_return + 1.96 * returns_std * np.sqrt(21))
            }
        else:
            confidence_interval = {"lower": predicted_return * 0.9, "upper": predicted_return * 1.1}
        
        return PerformancePredictionResponse(
            ticker=request.ticker,
            predicted_return=predicted_return,
            confidence_interval=confidence_interval,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )


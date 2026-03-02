"""
ARLN Configuration Module
Philosophical Shift: Configuration as Neural Synapse Weights
Each parameter tunes the cognitive architecture's response to market stimuli.
"""
import os
from dataclasses import dataclass
from typing import List, Optional
import logging

@dataclass
class DataSourceConfig:
    """Configuration for market data ingestion sources"""
    ccxt_exchange: str = os.getenv("CCXT_EXCHANGE", "binance")
    default_symbols: List[str] = os.getenv("DEFAULT_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "1h")
    yahoo_max_workers: int = int(os.getenv("YAHOO_MAX_WORKERS", "5"))
    
@dataclass
class FeatureEngineeringConfig:
    """Configuration for technical feature generation"""
    # Technical Indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Volatility Features
    atr_period: int = 14
    
    # Volume Features
    volume_ma_periods: List[int] = (5, 20, 50)
    
    # Time-based Features
    encode_time_features: bool = True

@dataclass
class FirebaseConfig:
    """Firebase state management configuration"""
    service_account_path: str = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "./serviceAccountKey.json")
    collection_name: str = "arln_market_state"
    max_retention_days: int = int(os.getenv("MAX_RETENTION_DAYS", "30"))
    
    def validate(self) -> bool:
        """Validate Firebase configuration exists"""
        if not os.path.exists(self.service_account_path):
            logging.error(f"Firebase service account not found at {self.service_account_path}")
            return False
        return True

@dataclass
class ARLNConfig:
    """Master configuration for ARLN Cognitive Architecture"""
    data_sources: DataSourceConfig = DataSourceConfig()
    features: FeatureEngineeringConfig = FeatureEngineeringConfig()
    firebase: FirebaseConfig = FirebaseConfig()
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize logging and validate critical dependencies"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Critical: Validate Firebase configuration
        if not self.firebase.validate():
            raise RuntimeError("Firebase configuration invalid. ARLN cannot operate without state management.")

# Global configuration instance
config = ARLNConfig()
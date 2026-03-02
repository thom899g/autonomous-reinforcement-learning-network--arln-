"""
ARLN Perceptual System: Multi-modal Data Fusion Engine
Architectural Philosophy: Market Consciousness through unified data representation
Edge Cases Handled:
1. Partial data source failures
2. Rate limiting and API throttling
3. Data schema mismatches
4. Network timeouts with exponential backoff
"""
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import firebase_admin
from firebase_admin import credentials, firestore
from tqdm import tqdm

from arln_config import config

logger = logging.getLogger(__name__)

class DataSourceFailure(Exception):
    """Custom exception for data source failures with recovery metadata"""
    def __init__(self, source: str, reason: str, recoverable: bool = True):
        self.source = source
        self.reason = reason
        self.recoverable = recoverable
        super().__init__(f"DataSource {source} failed: {reason}")

class MultiModalDataFusionEngine:
    """
    Core perceptual system for ARLN. Fuses multiple market data sources into
    a unified cognitive representation with built-in fault tolerance.
    """
    
    def __init__(self):
        """Initialize data fusion engine with Firebase state management"""
        logger.info("Initializing ARLN Perceptual System: Data Fusion Engine")
        
        # Initialize Firebase Firestore (CRITICAL for state management)
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.firebase.service_account_path)
                firebase_admin.initialize_app(cred)
            self.firestore_client = firestore.client()
            self.state_collection = self.firestore_client.collection(config.firebase.collection_name)
            logger.info("Firebase Firestore state management initialized")
        except Exception as e:
            logger.critical(f"Firebase initialization failed: {e}")
            raise RuntimeError("ARLN cannot operate without Firebase state management")
        
        # Initialize data sources with error resilience
        self.exchange = self._init_ccxt_exchange()
        self.data_sources_available = {
            'ccxt': self.exchange is not None,
            'yfinance': self._test_yfinance_connectivity()
        }
        logger.info(f"Available data sources: {self.data_sources_available}")
        
        # State tracking for adaptive learning
        self.last_successful_fusion = None
        self.consecutive_failures = 0
        self.performance_metrics = {
            'fusion_count': 0,
            'avg_processing_time': 0,
            'source_reliability': {k: 1.0 for k in self.data_sources_available}
        }
    
    def _init_ccxt_exchange(self) -> Optional[ccxt.Exchange]:
        """Initialize CCXT exchange with rate limiting and error handling"""
        try:
            exchange_class = getattr(ccxt, config.data_sources.ccxt_exchange)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'rateLimit': 1000  # Conservative rate limiting
            })
            # Test connectivity
            exchange.load_markets()
            logger.info(f"CCXT exchange {config.data_sources.ccxt_exchange} initialized successfully")
            return exchange
        except (ccxt.NetworkError, ccxt.ExchangeError, AttributeError) as e:
            logger.warning(f"CCXT exchange {config.data_sources.ccxt_exchange} initialization failed: {e}")
            return None
    
    def _test_yfinance_connectivity(self) -> bool:
        """Test Yahoo Finance connectivity with timeout"""
        try:
            # Quick test with a known symbol
            test_data =
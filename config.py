"""
Configuration file for StockStream Pro
Contains all configurable settings and constants
"""

import os
from datetime import date, timedelta

# Application Settings
APP_TITLE = "Stock Analyzer"
APP_ICON = "📈"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Default Date Settings
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date.today() + timedelta(days=1)  # Include today's data
MAX_PREDICTION_YEARS = 5
MIN_PREDICTION_YEARS = 1

# Chart Settings
CHART_HEIGHT = 600
TECHNICAL_ANALYSIS_HEIGHT = 800
CANDLESTICK_HEIGHT = 600

# Technical Analysis Settings
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# Prophet Settings
DEFAULT_CONFIDENCE_INTERVAL = 80
MIN_CONFIDENCE_INTERVAL = 80
MAX_CONFIDENCE_INTERVAL = 95

# Data Settings
CACHE_DIR = "/tmp"
YFINANCE_CACHE_DIR = os.environ.get("YFINANCE_CACHE_DIR", CACHE_DIR)

# UI Settings
METRIC_COLUMNS = 4
CHART_COLUMNS = 2

# Color Scheme
COLORS = {
    'primary': '#1f77b4',
    'positive': '#00ff00',
    'negative': '#ff0000',
    'neutral': '#808080',
    'background': '#f0f2f6'
}

# Sample Market Data (for dashboard)
SAMPLE_MARKET_DATA = {
    'market_cap': '₹45.2T',
    'market_cap_change': '2.3%',
    'sp500': '4,567.89',
    'sp500_change': '1.2%',
    'nasdaq': '14,234.56',
    'nasdaq_change': '0.8%',
    'nifty50': '24,567.89',
    'nifty50_change': '1.5%'
}

# Error Messages
ERROR_MESSAGES = {
    'no_data': 'No data found for the selected stock(s)',
    'invalid_dates': 'Start date must be before end date',
    'no_stock_selected': 'Please select at least one stock',
    'file_not_found': 'Required data file not found',
    'network_error': 'Network error while fetching data',
    'prediction_error': 'Error generating prediction'
}

# Success Messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Data loaded successfully',
    'prediction_generated': 'Prediction generated successfully',
    'analysis_complete': 'Analysis completed successfully'
}

# File Paths
DATA_FILE = 'StockStreamTickersData.csv'
LOGO_FILE = 'Images/StockStreamLogo1.png'

# API Settings
YAHOO_FINANCE_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1

# Performance Settings
CACHE_DURATION = 300  # 5 minutes
MAX_CACHE_SIZE = 100  # Maximum number of cached items

# Feature Flags
ENABLE_PORTFOLIO_ANALYSIS = True
ENABLE_TECHNICAL_ANALYSIS = True
ENABLE_PREDICTION = True
ENABLE_COMPARISON = True
ENABLE_DASHBOARD = True

# Debug Settings
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'
VERBOSE_LOGGING = os.environ.get('VERBOSE', 'False').lower() == 'true'
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
import yfinance as yf
import ta
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Page configuration b
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Formatting helper functions
# -----------------------------
def format_market_cap(value):
    """Return human-readable market cap with suffix (T, B, M, K)."""
    try:
        num = float(value or 0)
    except (TypeError, ValueError):
        return "N/A"

    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"₹{num/1_000_000_000_000:.2f}T"
    if abs_num >= 1_000_000_000:
        return f"₹{num/1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"₹{num/1_000_000:.2f}M"
    if abs_num >= 1_000:
        return f"₹{num/1_000:.2f}K"
    return f"₹{num:,.0f}"

def format_pe_ratio(value):
    """Format P/E ratio to two decimal places, handle missing values."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"

def format_dividend_yield(dividend_yield_value, dividend_rate=None, current_price=None):
    """Format dividend yield as percentage using robust logic.

    Precedence:
    1) If dividend_rate and current_price are valid (>0), compute yield = dividend_rate/current_price*100
    2) Else, use dividend_yield_value. If <1, treat as fraction; if >=1 and <= 100, treat as percent.
    3) Clamp any out-of-range values and return "N/A" on invalid inputs.
    """
    # 1) Compute from rate / price if available
    try:
        if dividend_rate is not None and current_price is not None:
            dr = float(dividend_rate)
            cp = float(current_price)
            if cp > 0 and dr >= 0:
                computed = (dr / cp) * 100.0
                if 0 <= computed <= 1000:  # sanity clamp
                    return f"{computed:.2f}%"
    except (TypeError, ValueError):
        pass

    # 2) Fallback to provided dividend_yield_value
    try:
        dy = float(dividend_yield_value)
    except (TypeError, ValueError):
        return "N/A"

    if dy < 0:
        return "N/A"

    # Heuristic: common cases
    # - dy in [0,1): treat as fraction -> multiply by 100
    # - dy in [1, 100]: treat as percent already
    # - dy > 100: likely already percent but erroneous; clamp to show at most two decimals without symbol inflation
    if dy < 1:
        dy_percent = dy * 100.0
    else:
        dy_percent = dy

    # Final sanity cap to avoid absurd values displaying due to bad data
    if dy_percent > 1000:
        return "N/A"

    return f"{dy_percent:.2f}%"

class StockAnalyzer:
    def __init__(self):
        self.ticker_data = self.load_ticker_data()
    
    def load_ticker_data(self):
        """Load stock ticker data from CSV"""
        try:
            # Auto-detect delimiter (handles tab- or comma-separated files)
            df = pd.read_csv("StockStreamTickersData.csv", sep=None, engine="python")
            # Normalize column names (strip spaces/case)
            df.columns = [c.strip() for c in df.columns]

            # Standardize expected columns
            rename_map = {}
            for col in df.columns:
                low = col.lower().replace("_", " ")
                if low in {"company name", "company"}:
                    rename_map[col] = "Company Name"
                elif low in {"symbol", "ticker", "symbols"}:
                    rename_map[col] = "Symbol"
            if rename_map:
                df = df.rename(columns=rename_map)

            # Validate required columns
            required_cols = {"Company Name", "Symbol"}
            if not required_cols.issubset(set(df.columns)):
                st.error("Ticker CSV must have 'Company Name' and 'Symbol' columns.")
                return pd.DataFrame()

            return df
        except FileNotFoundError:
            st.error("StockStreamTickersData.csv not found. Please ensure the file exists.")
            return pd.DataFrame()
    
    def refresh_ticker_data(self):
        """Refresh ticker data from CSV file"""
        self.ticker_data = self.load_ticker_data()
    
    def get_stock_data(self, symbol, start_date, end_date):
        """Fetch data for a symbol.

        Supported formats:
        - Regular tickers (e.g., RELIANCE.NS) via Yahoo Finance.
        - Indian mutual funds as MF:<scheme_code> via mfapi.in NAV history.
        """
        try:
            # Mutual Fund scheme codes (India): MF:<scheme_code>
            if isinstance(symbol, str) and symbol.startswith("MF:"):
                scheme_code = symbol.split("MF:", 1)[1].strip()
                return self.get_mutual_fund_nav_history(scheme_code, start_date, end_date)

            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
            
            # If single stock, ensure we have single-level columns
            if isinstance(symbol, str):
                # For single stock, flatten multi-level columns if they exist
                if isinstance(data.columns, pd.MultiIndex):
                    # Take the first (and only) stock's data
                    if len(data.columns.levels[1]) == 1:
                        stock_name = data.columns.levels[1][0]
                        data = data.xs(stock_name, level=1, axis=1)
                    else:
                        # Multiple stocks case - return as is
                        pass
                else:
                    # Already single-level columns
                    pass
            
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_mutual_fund_nav_history(self, scheme_code, start_date, end_date):
        """Fetch Indian mutual fund NAV history and map into OHLC-like dataframe.

        Source: mfapi.in (community API). Returns daily NAV values with dates.
        """
        try:
            import requests

            url = f"https://api.mfapi.in/mf/{scheme_code}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            payload = resp.json()

            data = payload.get("data", [])
            if not data:
                st.error(f"No NAV data found for MF:{scheme_code}")
                return None

            df = pd.DataFrame(data)
            # Expected keys: "date" (DD-MM-YYYY) and "nav" (string float)
            if "date" not in df.columns or "nav" not in df.columns:
                st.error(f"Unexpected NAV response format for MF:{scheme_code}")
                return None

            df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["date", "nav"]).sort_values("date")
            df = df.set_index("date")

            # Filter date range
            df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            if df.empty:
                st.error(f"No NAV data in the selected date range for MF:{scheme_code}")
                return None

            # Map NAV into OHLC schema to keep downstream indicators working.
            out = pd.DataFrame(index=df.index)
            out["Open"] = df["nav"]
            out["High"] = df["nav"]
            out["Low"] = df["nav"]
            out["Close"] = df["nav"]
            out["Adj Close"] = df["nav"]
            out["Volume"] = 0
            return out
        except Exception as e:
            st.error(f"Error fetching NAV for MF:{scheme_code}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or data.empty:
            return data
        
        try:
            # Handle multi-level columns (when multiple stocks are fetched)
            if isinstance(data.columns, pd.MultiIndex):
                # For single stock analysis, use the first (and only) stock
                if len(data.columns.levels[1]) == 1:
                    symbol = data.columns.levels[1][0]
                    close_data = data[('Close', symbol)]
                    volume_data = data[('Volume', symbol)]
                else:
                    # For multiple stocks, return original data without indicators
                    return data
            else:
                # Single-level columns (single stock)
                close_data = data['Close']
                volume_data = data['Volume']
            
            # Add technical indicators
            data['SMA_20'] = ta.trend.sma_indicator(close_data, window=20)
            data['SMA_50'] = ta.trend.sma_indicator(close_data, window=50)
            data['RSI'] = ta.momentum.rsi(close_data, window=14)
            data['MACD'] = ta.trend.macd(close_data)
            data['BB_upper'] = ta.volatility.bollinger_hband(close_data)
            data['BB_lower'] = ta.volatility.bollinger_lband(close_data)
            
            return data
        except Exception as e:
            st.warning(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def calculate_returns(self, data):
        """Calculate relative returns"""
        if data is None or data.empty:
            return data
        
        rel = data.pct_change()
        cumret = (1 + rel).cumprod() - 1
        return cumret.fillna(0)
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            # Mutual Fund scheme codes (India): MF:<scheme_code>
            if isinstance(symbol, str) and symbol.startswith("MF:"):
                scheme_code = symbol.split("MF:", 1)[1].strip()
                mf = self.get_mutual_fund_current_nav(scheme_code)
                if not mf:
                    return None
                return {
                    "name": mf.get("name", f"MF:{scheme_code}"),
                    "sector": "N/A",
                    "industry": "N/A",
                    "market_cap": 0,
                    "pe_ratio": "N/A",
                    "dividend_yield": 0,
                    "dividend_rate": 0,
                    "current_price": mf.get("current_price", 0),
                }

            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'current_price': info.get('currentPrice', 0)
            }
        except Exception as e:
            st.error(f"Error fetching info for {symbol}: {str(e)}")
            return None

    def get_mutual_fund_current_nav(self, scheme_code):
        """Fetch latest NAV for an Indian mutual fund scheme code (as 'current price')."""
        try:
            import requests

            url = f"https://api.mfapi.in/mf/{scheme_code}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            payload = resp.json()

            meta = payload.get("meta", {}) or {}
            scheme_name = meta.get("scheme_name") or meta.get("schemeName") or ""

            data = payload.get("data", [])
            if not data:
                return None

            df = pd.DataFrame(data)
            if "date" not in df.columns or "nav" not in df.columns:
                return None

            df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["date", "nav"]).sort_values("date")
            if df.empty:
                return None

            latest_nav = float(df["nav"].iloc[-1])
            return {"name": scheme_name.strip() or f"MF:{scheme_code}", "current_price": latest_nav}
        except Exception:
            return None
    
    def get_market_overview(self):
        """Get real-time market overview data"""
        try:
            # Get current date
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            
            # Fetch major indices data
            sp500 = yf.Ticker("^GSPC")
            nasdaq = yf.Ticker("^IXIC")
            nifty50 = yf.Ticker("^NSEI")
            sensex = yf.Ticker("^BSESN")
            
            # Get current and previous day data
            sp500_data = sp500.history(period="2d")
            nasdaq_data = nasdaq.history(period="2d")
            nifty50_data = nifty50.history(period="2d")
            sensex_data = sensex.history(period="2d")
            
            # Calculate changes
            def calculate_change(current, previous):
                if previous == 0:
                    return "0.00%"
                change = ((current - previous) / previous) * 100
                return f"{change:+.2f}%"
            
            # S&P 500
            sp500_current = sp500_data['Close'].iloc[-1]
            sp500_previous = sp500_data['Close'].iloc[-2] if len(sp500_data) > 1 else sp500_current
            sp500_change = calculate_change(sp500_current, sp500_previous)
            
            # NASDAQ
            nasdaq_current = nasdaq_data['Close'].iloc[-1]
            nasdaq_previous = nasdaq_data['Close'].iloc[-2] if len(nasdaq_data) > 1 else nasdaq_current
            nasdaq_change = calculate_change(nasdaq_current, nasdaq_previous)
            
            # Nifty 50
            nifty50_current = nifty50_data['Close'].iloc[-1]
            nifty50_previous = nifty50_data['Close'].iloc[-2] if len(nifty50_data) > 1 else nifty50_current
            nifty50_change = calculate_change(nifty50_current, nifty50_previous)
            
            # Sensex
            sensex_current = sensex_data['Close'].iloc[-1]
            sensex_previous = sensex_data['Close'].iloc[-2] if len(sensex_data) > 1 else sensex_current
            sensex_change = calculate_change(sensex_current, sensex_previous)
            
            return {
                'sp500': f"{sp500_current:,.2f}",
                'sp500_change': sp500_change,
                'nasdaq': f"{nasdaq_current:,.2f}",
                'nasdaq_change': nasdaq_change,
                'nifty50': f"{nifty50_current:,.2f}",
                'nifty50_change': nifty50_change,
                'sensex': f"{sensex_current:,.2f}",
                'sensex_change': sensex_change
            }
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            # Return sample data as fallback
            return {
                'sp500': '4,567.89',
                'sp500_change': '1.2%',
                'nasdaq': '14,234.56',
                'nasdaq_change': '0.8%',
                'nifty50': '24,567.89',
                'nifty50_change': '1.5%',
                'sensex': '73,000.00',
                'sensex_change': '0.9%'
            }
    
    def get_daily_performance(self, limit=5):
        """Get daily performance data for top and worst performing stocks"""
        try:
            if self.ticker_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get a sample of stocks for daily performance calculation
            sample_stocks = self.ticker_data.head(30)  # Limit to first 30 for performance
            symbols = sample_stocks['Symbol'].tolist()
            
            # Fetch a longer recent window to find two valid closes
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=14)
            
            data = self.get_stock_data(symbols, start_date, end_date)
            if data is None or data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Calculate daily performance
            performance_data = []
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close']
            else:
                close_data = data['Close']
            
            for idx, row in sample_stocks.iterrows():
                symbol = row['Symbol']
                company_name = row['Company Name']
                
                if symbol in close_data.columns:
                    try:
                        # Use the last two valid closes to avoid NaNs
                        series = close_data[symbol].dropna()
                        if len(series) < 2:
                            continue
                        current_price = series.iloc[-1]
                        previous_price = series.iloc[-2]
                        if previous_price == 0 or pd.isna(previous_price):
                            continue
                        
                        # Calculate daily change
                        daily_change = ((current_price - previous_price) / previous_price) * 100
                        
                        performance_data.append({
                            'Company Name': company_name,
                            'Symbol': symbol,
                            'Current Price': f"₹{current_price:.2f}",
                            'Daily Change (%)': float(daily_change),
                            'Daily Change': f"{daily_change:+.2f}%"
                        })
                    except Exception:
                        continue
            
            # Create DataFrame and sort by daily change
            performance_df = pd.DataFrame(performance_data)
            if performance_df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Sort by daily change
            performance_df = performance_df.sort_values('Daily Change (%)', ascending=False)
            
            # Get top and worst performers
            top_performers = performance_df.head(limit)
            worst_performers = performance_df.tail(limit).sort_values('Daily Change (%)', ascending=True)
            
            return top_performers, worst_performers
            
        except Exception as e:
            st.error(f"Error calculating daily performance: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def get_top_performers(self, start_date, end_date, limit=10):
        """Get top performing stocks from the dataset"""
        try:
            if self.ticker_data.empty:
                return pd.DataFrame()
            
            # Get a sample of stocks for performance calculation
            sample_stocks = self.ticker_data.head(20)  # Limit to first 20 for performance
            symbols = sample_stocks['Symbol'].tolist()
            
            # Fetch data for all symbols
            data = self.get_stock_data(symbols, start_date, end_date)
            if data is None or data.empty:
                return self.ticker_data.head(limit)
            
            # Calculate returns
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                adj_close_data = data['Adj Close']
            else:
                adj_close_data = data['Adj Close']
            returns = self.calculate_returns(adj_close_data)
            
            # Calculate total returns for each stock
            performance_data = []
            for idx, row in sample_stocks.iterrows():
                symbol = row['Symbol']
                company_name = row['Company Name']
                
                if symbol in returns.columns:
                    total_return = returns[symbol].iloc[-1] * 100
                    current_price = data['Adj Close'][symbol].iloc[-1]
                    performance_data.append({
                        'Company Name': company_name,
                        'Symbol': symbol,
                        'Total Return (%)': f"{total_return:.2f}",
                        'Current Price': f"₹{current_price:.2f}"
                    })
            
            # Sort by total return and return top performers
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Total Return (%)', ascending=False)
            
            return performance_df.head(limit)
            
        except Exception as e:
            st.error(f"Error calculating top performers: {str(e)}")
            return self.ticker_data.head(limit)

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer</h1>', unsafe_allow_html=True)
    
    analyzer = StockAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        selected = option_menu(
            "Main Menu",
            [
                "🏠 Dashboard",
                "🔍 Price Analysis",
                "🔮 Price Prediction",
                "📊 Portfolio Analysis",
                "ℹ️ About"
            ],
            icons=['house', 'search', 'crystal-ball', 'pie-chart', 'info-circle'],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        start_date = st.date_input('Start Date', datetime.date(2020, 1, 1))
        end_date = st.date_input('End Date', datetime.date.today())
        
        # Add refresh button
        if st.button('🔄 Refresh Data'):
            analyzer.refresh_ticker_data()
            st.success('Data refreshed successfully!')
        
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
    
    # Main content based on selection
    if selected == "🏠 Dashboard":
        show_dashboard(analyzer, start_date, end_date)
    elif selected == "🔍 Price Analysis":
        show_stock_analysis(analyzer, start_date, end_date)
    elif selected == "🔮 Price Prediction":
        show_stock_prediction(analyzer, start_date, end_date)
    elif selected == "📊 Portfolio Analysis":
        show_portfolio_analysis(analyzer, start_date, end_date)
    elif selected == "ℹ️ About":
        show_about()

def show_dashboard(analyzer, start_date, end_date):
    """Main dashboard with market overview"""
    st.header("📊 Market Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch real-time market data
    with st.spinner('Loading market data...'):
        try:
            # Get current market data
            market_data = analyzer.get_market_overview()
            
            with col1:
                st.metric("S&P 500", market_data['sp500'], market_data['sp500_change'])
            with col2:
                st.metric("NASDAQ", market_data['nasdaq'], market_data['nasdaq_change'])
            with col3:
                st.metric("Nifty 50", market_data['nifty50'], market_data['nifty50_change'])
            with col4:
                st.metric("Sensex", market_data['sensex'], market_data['sensex_change'])
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
            # Fallback to sample data
            with col1:
                st.metric("S&P 500", "4,567.89", "1.2%")
            with col2:
                st.metric("NASDAQ", "14,234.56", "0.8%")
            with col3:
                st.metric("Nifty 50", "24,567.89", "1.5%")
            with col4:
                st.metric("Sensex", "73,000.00", "0.9%")
    
    st.markdown("---")
    
    # Daily Performance Section
    st.subheader("📊 Daily Stock Performance")
    
    if not analyzer.ticker_data.empty:
        with st.spinner('Calculating daily performance...'):
            try:
                top_performers, worst_performers = analyzer.get_daily_performance(5)
                
                # Stack: Top first, then Worst below
                st.markdown("### 🚀 Top 5 Performers Today")
                if not top_performers.empty:
                    display_top = top_performers[['Company Name', 'Symbol', 'Current Price', 'Daily Change']].copy()
                    st.dataframe(display_top, width='stretch', hide_index=True)
                else:
                    st.info("No data available for top performers")
                
                st.markdown("### 📉 Worst 5 Performers Today")
                if not worst_performers.empty:
                    display_worst = worst_performers[['Company Name', 'Symbol', 'Current Price', 'Daily Change']].copy()
                    st.dataframe(display_worst, width='stretch', hide_index=True)
                else:
                    st.info("No data available for worst performers")
                        
            except Exception as e:
                st.error(f"Error calculating daily performance: {str(e)}")
    

## Stock comparison feature removed as per request

def show_stock_analysis(analyzer, start_date, end_date):
    """Detailed price analysis of a single asset"""
    st.header("🔍 Individual Price Analysis")
    
    if analyzer.ticker_data.empty:
        st.error("No ticker data available")
        return

    # Build lookup once
    symbol_dict = dict(zip(analyzer.ticker_data["Company Name"], analyzer.ticker_data["Symbol"]))
    all_names = analyzer.ticker_data["Company Name"].tolist()

    # Smooth selection UX: use a form so typing/clearing doesn't rerun the app on every keypress
    with st.form("price_analysis_picker"):
        query = st.text_input("Search asset name (stocks + mutual funds)", value="", placeholder="Type to search…")

        # Filter client-side and cap results for responsiveness
        q = query.strip().lower()
        if q:
            filtered = [n for n in all_names if q in n.lower()]
        else:
            filtered = all_names
        if len(filtered) > 300:
            filtered = filtered[:300]
            st.caption("Showing first 300 matches. Narrow your search for more.")

        selected_stock = st.selectbox("Select an asset", filtered, index=0 if filtered else None)
        submitted = st.form_submit_button("Analyze Price")

    if not submitted:
        return

    if not selected_stock:
        st.warning("Please select an asset")
        return

    symbol = symbol_dict.get(selected_stock)
    if not symbol:
        st.error("Could not resolve symbol for the selected asset.")
        return

    with st.spinner('Loading asset data...'):
        # Fetch data
        data = analyzer.get_stock_data(symbol, start_date, end_date)
        if data is None:
            return
        
        # Add technical indicators
        data = analyzer.calculate_technical_indicators(data)
        
        # Asset info
        info = analyzer.get_stock_info(symbol)
        if info:
            # Mutual funds: only show "Current Price" (latest NAV). Skip stock-only fields.
            if isinstance(symbol, str) and symbol.startswith("MF:"):
                st.metric("Current Price (NAV)", f"₹{info.get('current_price', 0):.2f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{info['current_price']:.2f}")
                with col2:
                    st.metric("Market Cap", format_market_cap(info['market_cap']))
                with col3:
                    st.metric("P/E Ratio", format_pe_ratio(info['pe_ratio']))
                with col4:
                    st.metric(
                        "Dividend Yield",
                        format_dividend_yield(
                            info.get('dividend_yield'),
                            dividend_rate=info.get('dividend_rate'),
                            current_price=info.get('current_price')
                        )
                    )

        # Charts: show three charts with minimal modebar and dark theme
        config = {
            'modeBarButtonsToRemove': [
                'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage',
                'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d',
                'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGl2d', 'hoverClosestPie'
            ],
            'displaylogo': False
        }

        try:
            # Ensure we have the required columns
            if 'Close' not in data.columns:
                st.error("Close price data not available")
                return
            
            # Clean data - remove any NaN values
            clean_data = data.dropna(subset=['Close'])
            if clean_data.empty:
                st.error("No valid price data available")
                return

            # 1) Historical Price Chart
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=clean_data.index, 
                y=clean_data['Close'], 
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            price_fig.update_layout(
                title=f'{selected_stock} - Historical Price',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=400,
                template='plotly_dark',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified'
            )
            st.plotly_chart(price_fig, use_container_width=True, config=config)

            # 2) Moving Averages Chart
            ma_fig = go.Figure()
            ma_fig.add_trace(go.Scatter(
                x=clean_data.index, 
                y=clean_data['Close'], 
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add moving averages if they exist and have valid data
            if 'SMA_20' in clean_data.columns and not clean_data['SMA_20'].isna().all():
                ma_fig.add_trace(go.Scatter(
                    x=clean_data.index, 
                    y=clean_data['SMA_20'], 
                    name='SMA 20',
                    line=dict(color='#ff7f0e', width=1.5)
                ))
            
            if 'SMA_50' in clean_data.columns and not clean_data['SMA_50'].isna().all():
                ma_fig.add_trace(go.Scatter(
                    x=clean_data.index, 
                    y=clean_data['SMA_50'], 
                    name='SMA 50',
                    line=dict(color='#2ca02c', width=1.5)
                ))
            
            ma_fig.update_layout(
                title=f'{selected_stock} - Moving Averages',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=400,
                template='plotly_dark',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified'
            )
            st.plotly_chart(ma_fig, use_container_width=True, config=config)

            # 3) Volume Chart
            # Mutual funds: skip volume chart
            if isinstance(symbol, str) and symbol.startswith("MF:"):
                pass
            elif 'Volume' in clean_data.columns and not clean_data['Volume'].isna().all():
                vol_fig = go.Figure()
                vol_fig.add_trace(go.Bar(
                    x=clean_data.index, 
                    y=clean_data['Volume'], 
                    name='Volume',
                    marker_color='#17becf'
                ))
                vol_fig.update_layout(
                    title=f'{selected_stock} - Volume',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=300,
                    template='plotly_dark',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    hovermode='x unified'
                )
                st.plotly_chart(vol_fig, use_container_width=True, config=config)
            else:
                st.warning("Volume data not available for this stock")

        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
            st.error("Please try selecting a different stock or check your data connection.")

def show_stock_prediction(analyzer, start_date, end_date):
    """Price prediction using Prophet"""
    st.header("🔮 Price Prediction")
    
    if analyzer.ticker_data.empty:
        st.error("No ticker data available")
        return
    
    # Stock selection
    tickers = analyzer.ticker_data["Company Name"].tolist()
    selected_stock = st.selectbox('Select a stock for prediction:', tickers)
    
    if not selected_stock:
        st.warning("Please select a stock")
        return
    
    symbol_dict = dict(zip(analyzer.ticker_data["Company Name"], analyzer.ticker_data["Symbol"]))
    symbol = symbol_dict[selected_stock]
    
    # Prediction period
    col1, col2 = st.columns(2)
    with col1:
        years = st.slider('Years to predict:', 1, 5, 1)
    with col2:
        confidence_interval = st.slider('Confidence Interval:', 80, 95, 80)
    
    if st.button('Generate Prediction'):
        with st.spinner('Generating prediction...'):
            try:
                # Fetch data
                data = analyzer.get_stock_data(symbol, start_date, end_date)
                if data is None:
                    return
                
                # Check if Close column exists
                if 'Close' not in data.columns:
                    st.error("Close price data not available for prediction")
                    return
                
                # Clean data - remove any NaN values
                clean_data = data.dropna(subset=['Close'])
                if clean_data.empty:
                    st.error("No valid price data available for prediction")
                    return
                
                # Prepare data for Prophet
                df_prophet = clean_data[['Close']].reset_index()
                df_prophet.columns = ['ds', 'y']
                
                # Train Prophet model
                model = Prophet(interval_width=confidence_interval/100)
                model.fit(df_prophet)
                
                # Make future predictions
                future = model.make_future_dataframe(periods=years*365)
                forecast = model.predict(future)
                
                # Display results
                st.subheader(f'📊 Prediction for {selected_stock}')
                
                # Forecast plot
                fig = plot_plotly(model, forecast)
                fig.update_layout(
                    title=f'{selected_stock} - Price Forecast ({years} years)',
                    height=600,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.error("Please try selecting a different stock or check your data connection.")


def show_portfolio_analysis(analyzer, start_date, end_date):
    """Portfolio analysis and optimization"""
    st.header("📊 Portfolio Analysis")

    if analyzer.ticker_data.empty:
        st.error("No ticker data available")
        return

    st.caption("Optimize a long-only portfolio (weights sum to 1) using a Monte Carlo mean-variance search.")

    # --- Asset selection (smooth UX)
    symbol_dict = dict(zip(analyzer.ticker_data["Company Name"], analyzer.ticker_data["Symbol"]))
    all_names = analyzer.ticker_data["Company Name"].tolist()

    with st.form("portfolio_builder"):
        query = st.text_input(
            "Search assets to add (stocks + mutual funds)",
            value="",
            placeholder="Type to filter…",
        )
        q = query.strip().lower()
        if q:
            filtered = [n for n in all_names if q in n.lower()]
        else:
            filtered = all_names
        if len(filtered) > 300:
            filtered = filtered[:300]
            st.caption("Showing first 300 matches. Narrow your search for more.")

        selected_assets = st.multiselect("Select assets", options=filtered, default=[])

        colA, colB, colC = st.columns(3)
        with colA:
            objective = st.selectbox("Objective", ["Max Sharpe", "Min Volatility", "Max Return"], index=0)
        with colB:
            risk_free_rate = st.number_input("Risk-free rate (annual, %)", min_value=0.0, max_value=25.0, value=6.0, step=0.5)
        with colC:
            n_portfolios = st.slider("Random portfolios", min_value=2000, max_value=30000, value=12000, step=2000)

        colX, colY = st.columns(2)
        with colX:
            initial_capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=100000.0, step=1000.0)
        with colY:
            max_assets = st.slider("Max assets to optimize", min_value=2, max_value=20, value=10)

        run_opt = st.form_submit_button("Run Portfolio Optimization")

    if not run_opt:
        return

    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets to optimize.")
        return

    # Cap count for runtime / API stability
    selected_assets = selected_assets[:max_assets]
    symbols = [symbol_dict.get(n) for n in selected_assets]
    if any(s is None for s in symbols):
        st.error("One or more selected assets could not be resolved to a symbol.")
        return

    rf = float(risk_free_rate) / 100.0

    @st.cache_data(show_spinner=False)
    def _fetch_price_series(symbol, start_date_str, end_date_str):
        data = analyzer.get_stock_data(symbol, start_date_str, end_date_str)
        if data is None or getattr(data, "empty", True):
            return None
        if "Adj Close" in data.columns:
            s = data["Adj Close"].copy()
        elif "Close" in data.columns:
            s = data["Close"].copy()
        else:
            return None
        s = s.dropna()
        s.name = symbol
        return s

    with st.spinner("Fetching price history..."):
        series_list = []
        for sym in symbols:
            s = _fetch_price_series(sym, str(start_date), str(end_date))
            if s is not None and not s.empty:
                series_list.append(s)

    if len(series_list) < 2:
        st.error("Not enough valid price series found for optimization. Try different assets or a wider date range.")
        return

    prices = pd.concat(series_list, axis=1).sort_index()
    # Keep only rows where all assets have data (simple + robust)
    prices = prices.dropna(how="any")
    if len(prices) < 30:
        st.error("Not enough overlapping price history across selected assets (need ~30+ data points).")
        return

    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("Unable to compute returns for the selected assets.")
        return

    # Annualized stats
    ann_factor = 252
    mu = returns.mean() * ann_factor
    cov = returns.cov() * ann_factor

    # Monte Carlo portfolio search (long-only)
    np.random.seed(42)
    k = returns.shape[1]
    W = np.random.dirichlet(np.ones(k), size=int(n_portfolios))  # (n, k)
    port_ret = W @ mu.values
    port_var = np.einsum("ij,jk,ik->i", W, cov.values, W)
    port_vol = np.sqrt(np.maximum(port_var, 1e-12))
    port_sharpe = (port_ret - rf) / np.maximum(port_vol, 1e-12)

    if objective == "Max Sharpe":
        best_i = int(np.nanargmax(port_sharpe))
    elif objective == "Min Volatility":
        best_i = int(np.nanargmin(port_vol))
    else:  # Max Return
        best_i = int(np.nanargmax(port_ret))

    best_w = W[best_i]
    best_ret = float(port_ret[best_i])
    best_vol = float(port_vol[best_i])
    best_sharpe = float(port_sharpe[best_i])

    weights_df = pd.DataFrame(
        {
            "Asset": list(prices.columns),
            "Symbol": list(prices.columns),
            "Weight (%)": (best_w * 100.0),
        }
    ).sort_values("Weight (%)", ascending=False)
    weights_df["Weight (%)"] = weights_df["Weight (%)"].map(lambda x: float(f"{x:.2f}"))

    st.markdown("### ✅ Optimized Portfolio")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Expected Return (annual)", f"{best_ret*100:.2f}%")
    with m2:
        st.metric("Volatility (annual)", f"{best_vol*100:.2f}%")
    with m3:
        st.metric("Sharpe (vs RF)", f"{best_sharpe:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Allocation")
        pie = go.Figure(
            data=[
                go.Pie(
                    labels=weights_df["Asset"],
                    values=weights_df["Weight (%)"],
                    hole=0.35,
                )
            ]
        )
        pie.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(pie, use_container_width=True)
        st.dataframe(weights_df, width="stretch", hide_index=True)

    with col2:
        st.subheader("📈 Simulated Efficient Frontier")
        ef = go.Figure()
        ef.add_trace(
            go.Scatter(
                x=port_vol * 100.0,
                y=port_ret * 100.0,
                mode="markers",
                marker=dict(
                    size=4,
                    color=port_sharpe,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe"),
                ),
                name="Portfolios",
            )
        )
        ef.add_trace(
            go.Scatter(
                x=[best_vol * 100.0],
                y=[best_ret * 100.0],
                mode="markers",
                marker=dict(size=12, color="#ff4b4b", symbol="star"),
                name="Selected",
            )
        )
        ef.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Volatility (annual, %)",
            yaxis_title="Return (annual, %)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(ef, use_container_width=True)

    st.markdown("### 📉 Backtested Portfolio Value (weight-rebalanced daily)")
    daily_port_ret = (returns.values @ best_w)
    portfolio_value = pd.Series(initial_capital * (1.0 + daily_port_ret).cumprod(), index=returns.index, name="Portfolio Value")

    pv_fig = go.Figure()
    pv_fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values, name="Portfolio Value", line=dict(width=2)))
    pv_fig.update_layout(template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Value (₹)")
    st.plotly_chart(pv_fig, use_container_width=True)

def show_about():
    """About page"""
    st.header("ℹ️ About Stock Analyzer")
    
    st.markdown("""
    ## 🚀 Stock Analyzer
    
    **Stock Analyzer** is an advanced stock analysis and portfolio management application built with Streamlit. 
    It provides comprehensive tools for stock market analysis, prediction, and portfolio optimization.
    
    ### ✨ Features
    
    - **📊 Dashboard**: Market overview and key metrics
    - **📈 Stock Comparison**: Compare multiple assets side by side
    - **🔍 Price Analysis**: Detailed technical analysis with indicators
    - **🔮 Price Prediction**: AI-powered price forecasting using Prophet
    - **📊 Portfolio Analysis**: Portfolio optimization and risk analysis
    
    ### 🛠️ Built With
    
    - **Streamlit** - Web application framework
    - **Pandas** - Data manipulation and analysis
    - **Plotly** - Interactive visualizations
    - **Yahoo Finance** - Real-time stock data
    - **Prophet** - Time series forecasting
    - **Technical Analysis Library** - Technical indicators
    
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
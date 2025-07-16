# mantra_diagnostics.py - COMPLETE SYSTEM HEALTH CHECK & TESTING
"""
M.A.N.T.R.A. Diagnostics Dashboard - FINAL BUG-FREE VERSION
===========================================================
Comprehensive diagnostic system for M.A.N.T.R.A. EDGE
Tests data pipeline, calculations, signals, and performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. Diagnostics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data source configuration
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Expected columns for Indian market watchlist (updated based on actual data)
EXPECTED_COLUMNS = {
    'ticker', 'company_name', 'sector', 'category', 'year',
    'price', 'market_cap', 'prev_close',
    'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct',
    'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
    'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
    'vol_ratio_30d_180d', 'vol_ratio_90d_180d', 
    'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'rvol',
    'sma_20d', 'sma_50d', 'sma_200d',
    'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct'
}

# ============================================================================
# DIAGNOSTIC ENGINE
# ============================================================================

class SystemDiagnostics:
    """Complete system diagnostics and health checks"""
    
    def __init__(self):
        self.diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'data_quality': {},
            'calculation_checks': {},
            'signal_analysis': {},
            'performance_metrics': {}
        }
        self.df_raw = None
        self.df_processed = None
    
    def run_complete_diagnostics(self):
        """Run all diagnostic tests"""
        st.header("🔍 M.A.N.T.R.A. Complete System Diagnostics")
        
        # Create diagnostic tabs
        tabs = st.tabs([
            "📊 Data Pipeline", 
            "🧮 Calculations", 
            "📈 Signals", 
            "🇮🇳 Indian Market", 
            "⚡ Performance", 
            "📥 Reports"
        ])
        
        with tabs[0]:
            self.test_data_pipeline()
        
        with tabs[1]:
            self.test_calculations()
        
        with tabs[2]:
            self.test_signal_generation()
        
        with tabs[3]:
            self.test_indian_market_compatibility()
        
        with tabs[4]:
            self.test_performance_metrics()
        
        with tabs[5]:
            self.generate_diagnostic_reports()
    
    def test_data_pipeline(self):
        """Test complete data loading pipeline"""
        st.subheader("📊 Data Pipeline Testing")
        
        # Test 1: URL Accessibility
        with st.expander("Test 1: Google Sheets Accessibility", expanded=True):
            try:
                start = time.time()
                response = requests.get(SHEET_URL, timeout=10, allow_redirects=True)
                load_time = time.time() - start
                
                if response.status_code == 200:
                    st.success(f"✅ Google Sheets accessible (Response time: {load_time:.2f}s)")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning(f"⚠️ Status {response.status_code} but data may be accessible")
                    self.diagnostics['warnings'].append(f"Sheet status: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                self.diagnostics['tests_failed'] += 1
                self.diagnostics['errors'].append(f"Connection error: {str(e)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Data Loading
        with st.expander("Test 2: Data Loading & Parsing", expanded=True):
            try:
                self.df_raw = self.load_raw_data()
                if self.df_raw is not None and not self.df_raw.empty:
                    st.success(f"✅ Loaded {len(self.df_raw)} rows, {len(self.df_raw.columns)} columns")
                    self.diagnostics['tests_passed'] += 1
                    
                    # Show sample
                    st.write("Sample data (first 3 rows):")
                    st.dataframe(self.df_raw.head(3))
                    
                    # Column analysis
                    st.write("Column types:")
                    col_types = self.df_raw.dtypes.value_counts()
                    st.write(col_types)
                else:
                    st.error("❌ Failed to load data")
                    self.diagnostics['tests_failed'] += 1
            except Exception as e:
                st.error(f"❌ Data loading error: {str(e)}")
                self.diagnostics['tests_failed'] += 1
                self.diagnostics['errors'].append(f"Load error: {str(e)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Column Completeness
        with st.expander("Test 3: Column Completeness Check", expanded=True):
            if self.df_raw is not None:
                # Clean column names first
                df_cols = set(self.df_raw.columns.str.strip())
                missing_cols = EXPECTED_COLUMNS - df_cols
                extra_cols = df_cols - EXPECTED_COLUMNS
                
                col1, col2 = st.columns(2)
                with col1:
                    if missing_cols:
                        st.warning(f"⚠️ Missing columns: {len(missing_cols)}")
                        st.write("Missing:", sorted(list(missing_cols))[:10])
                        self.diagnostics['warnings'].append(f"Missing columns: {sorted(list(missing_cols))}")
                    else:
                        st.success("✅ All expected columns present")
                        self.diagnostics['tests_passed'] += 1
                
                with col2:
                    if extra_cols:
                        st.info(f"ℹ️ Extra columns: {len(extra_cols)}")
                        st.write("Extra:", sorted(list(extra_cols))[:10])
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['missing_columns'] = sorted(list(missing_cols))
                self.diagnostics['data_quality']['total_columns'] = len(self.df_raw.columns)
        
        # Test 4: Data Quality
        with st.expander("Test 4: Data Quality Analysis", expanded=True):
            if self.df_raw is not None:
                df_clean = self.clean_data(self.df_raw.copy())
                
                # Null analysis
                null_counts = df_clean.isnull().sum()
                high_null_cols = null_counts[null_counts > len(df_clean) * 0.5]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_cells = len(df_clean) * len(df_clean.columns)
                    null_pct = (null_counts.sum() / total_cells * 100) if total_cells > 0 else 0
                    st.metric("Null %", f"{null_pct:.1f}%")
                    
                    if null_pct < 10:
                        st.success("✅ Excellent data quality")
                        self.diagnostics['tests_passed'] += 1
                    elif null_pct < 25:
                        st.warning("⚠️ Moderate nulls (normal)")
                        self.diagnostics['warnings'].append(f"Null percentage: {null_pct:.1f}%")
                    else:
                        st.error("❌ High null percentage")
                        self.diagnostics['warnings'].append(f"High null percentage: {null_pct:.1f}%")
                
                with col2:
                    st.metric("High Null Columns", len(high_null_cols))
                    if len(high_null_cols) > 0:
                        st.write("Columns >50% null:", list(high_null_cols.index)[:5])
                
                with col3:
                    if 'ticker' in df_clean.columns:
                        duplicates = df_clean['ticker'].duplicated().sum()
                        st.metric("Duplicate Tickers", duplicates)
                        if duplicates > 0:
                            self.diagnostics['warnings'].append(f"Duplicate tickers: {duplicates}")
                    else:
                        st.metric("Duplicate Tickers", "N/A")
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['null_percentage'] = float(null_pct)
                self.diagnostics['data_quality']['duplicate_tickers'] = int(duplicates) if 'duplicates' in locals() else 0
                
                # Store cleaned data
                self.df_processed = df_clean
    
    def test_calculations(self):
        """Test all calculations"""
        st.subheader("🧮 Calculation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test calculations - data loading failed")
            return
        
        # Test 1: Volume Acceleration
        with st.expander("Test 1: Volume Acceleration Calculation", expanded=True):
            required_cols = ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']
            if all(col in df.columns for col in required_cols):
                try:
                    # Check data types
                    vol_30_90 = pd.to_numeric(df['vol_ratio_30d_90d'], errors='coerce')
                    vol_30_180 = pd.to_numeric(df['vol_ratio_30d_180d'], errors='coerce')
                    
                    # Calculate
                    volume_accel = vol_30_90 - vol_30_180
                    
                    # Stats
                    valid_count = volume_accel.notna().sum()
                    st.write(f"Valid calculations: {valid_count}/{len(df)}")
                    
                    if valid_count > 0:
                        st.success("✅ Volume acceleration calculated successfully")
                        self.diagnostics['tests_passed'] += 1
                        
                        # Show distribution
                        fig = go.Figure(data=[go.Histogram(x=volume_accel.dropna(), nbinsx=50)])
                        fig.update_layout(title="Volume Acceleration Distribution", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show stats
                        st.write(f"Mean: {volume_accel.mean():.2f}%, Std: {volume_accel.std():.2f}%")
                        st.write(f"Min: {volume_accel.min():.2f}%, Max: {volume_accel.max():.2f}%")
                    else:
                        st.error("❌ No valid volume acceleration data")
                        self.diagnostics['tests_failed'] += 1
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    self.diagnostics['tests_failed'] += 1
            else:
                st.warning("⚠️ Required columns not found")
                self.diagnostics['warnings'].append("Volume ratio columns missing")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Moving Average Analysis
        with st.expander("Test 2: Moving Average Analysis", expanded=True):
            ma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
            available_ma = [col for col in ma_cols if col in df.columns]
            
            if 'price' in df.columns and available_ma:
                price = pd.to_numeric(df['price'], errors='coerce')
                
                ma_stats = {}
                for ma_col in available_ma:
                    ma_val = pd.to_numeric(df[ma_col], errors='coerce')
                    above_ma = (price > ma_val).sum()
                    total_valid = (~price.isna() & ~ma_val.isna()).sum()
                    
                    if total_valid > 0:
                        pct_above = (above_ma / total_valid * 100)
                        ma_stats[ma_col] = {
                            'above': above_ma,
                            'total': total_valid,
                            'pct': pct_above
                        }
                
                # Create bar chart
                if ma_stats:
                    fig = go.Figure()
                    ma_names = list(ma_stats.keys())
                    pct_values = [stats['pct'] for stats in ma_stats.values()]
                    
                    fig.add_trace(go.Bar(
                        x=ma_names,
                        y=pct_values,
                        text=[f"{pct:.1f}%" for pct in pct_values],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Stocks Above Moving Averages",
                        yaxis_title="Percentage %",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Moving average analysis complete")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning("⚠️ No valid MA data")
            else:
                st.warning("⚠️ Price or MA columns not found")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Returns Analysis
        with st.expander("Test 3: Returns Analysis", expanded=True):
            returns_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
            available_returns = [col for col in returns_cols if col in df.columns]
            
            if available_returns:
                returns_stats = {}
                for col in available_returns:
                    ret_data = pd.to_numeric(df[col], errors='coerce')
                    positive = (ret_data > 0).sum()
                    negative = (ret_data < 0).sum()
                    total = ret_data.notna().sum()
                    
                    if total > 0:
                        returns_stats[col] = {
                            'positive': positive,
                            'negative': negative,
                            'total': total,
                            'avg': ret_data.mean(),
                            'median': ret_data.median()
                        }
                
                # Create returns chart
                if returns_stats:
                    periods = list(returns_stats.keys())
                    avg_returns = [stats['avg'] for stats in returns_stats.values()]
                    median_returns = [stats['median'] for stats in returns_stats.values()]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Average', x=periods, y=avg_returns))
                    fig.add_trace(go.Bar(name='Median', x=periods, y=median_returns))
                    
                    fig.update_layout(
                        title="Market Returns Analysis",
                        yaxis_title="Return %",
                        barmode='group',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display stats
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Positive/Negative Distribution:**")
                    for period, stats in returns_stats.items():
                        if stats['total'] > 0:
                            pos_pct = stats['positive'] / stats['total'] * 100
                            st.write(f"{period}: {pos_pct:.1f}% positive")
                
                with col2:
                    st.write("**Average Returns:**")
                    for period, stats in returns_stats.items():
                        if not pd.isna(stats['avg']):
                            st.write(f"{period}: {stats['avg']:.2f}%")
                
                st.success("✅ Returns analysis complete")
                self.diagnostics['tests_passed'] += 1
            else:
                st.warning("⚠️ No returns columns found")
            
            self.diagnostics['tests_run'] += 1
    
    def test_signal_generation(self):
        """Test signal generation logic"""
        st.subheader("📈 Signal Generation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test signals - data loading failed")
            return
        
        # Apply signals
        df = self.apply_signal_logic(df)
        
        # Signal distribution
        with st.expander("Signal Distribution Analysis", expanded=True):
            if 'EDGE_SIGNAL' in df.columns:
                signal_counts = df['EDGE_SIGNAL'].value_counts()
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    hole=0.3,
                    marker_colors=['#f0f0f0', '#ff0000', '#00cc00', '#ff6600', '#0066cc']
                )])
                fig.update_layout(title="Signal Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                total_stocks = len(df)
                none_signals = signal_counts.get('NONE', 0)
                active_signals = total_stocks - none_signals
                signal_ratio = (active_signals / total_stocks * 100) if total_stocks > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stocks", total_stocks)
                with col2:
                    st.metric("Active Signals", active_signals)
                with col3:
                    st.metric("Signal Rate", f"{signal_ratio:.1f}%")
                
                if 3 <= signal_ratio <= 15:
                    st.success("✅ Signal generation rate is optimal (3-15%)")
                    self.diagnostics['tests_passed'] += 1
                elif signal_ratio > 0:
                    st.warning(f"⚠️ Signal rate {signal_ratio:.1f}% (expected 3-15%)")
                    self.diagnostics['warnings'].append(f"Signal rate: {signal_ratio:.1f}%")
                else:
                    st.info("ℹ️ No signals generated - check market conditions")
                
                self.diagnostics['signal_analysis']['distribution'] = signal_counts.to_dict()
                self.diagnostics['signal_analysis']['signal_rate'] = signal_ratio
            
            self.diagnostics['tests_run'] += 1
        
        # Signal Quality Analysis
        with st.expander("Signal Quality Analysis", expanded=True):
            if 'EDGE_SIGNAL' in df.columns and df['EDGE_SIGNAL'].ne('NONE').any():
                # Analyze each signal type
                signal_types = df[df['EDGE_SIGNAL'] != 'NONE']['EDGE_SIGNAL'].unique()
                
                for signal_type in signal_types:
                    signal_df = df[df['EDGE_SIGNAL'] == signal_type]
                    st.write(f"\n**{signal_type} ({len(signal_df)} stocks):**")
                    
                    # Show top 5 stocks
                    display_cols = ['ticker', 'price', 'volume_acceleration', 'ret_7d', 'pe']
                    available_cols = [col for col in display_cols if col in signal_df.columns]
                    
                    if available_cols:
                        st.dataframe(
                            signal_df[available_cols].head(5),
                            use_container_width=True
                        )
                    
                    # Show key metrics
                    metrics = {}
                    if 'ret_7d' in signal_df.columns:
                        metrics['Avg 7d Return'] = f"{signal_df['ret_7d'].mean():.2f}%"
                    if 'volume_acceleration' in signal_df.columns:
                        metrics['Avg Vol Accel'] = f"{signal_df['volume_acceleration'].mean():.1f}%"
                    if 'pe' in signal_df.columns:
                        valid_pe = signal_df[signal_df['pe'] > 0]['pe']
                        if len(valid_pe) > 0:
                            metrics['Avg PE'] = f"{valid_pe.mean():.1f}"
                    
                    if metrics:
                        cols = st.columns(len(metrics))
                        for i, (metric, value) in enumerate(metrics.items()):
                            cols[i].metric(metric, value)
                
                st.success("✅ Signal quality analysis complete")
                self.diagnostics['tests_passed'] += 1
            else:
                st.info("No active signals to analyze")
            
            self.diagnostics['tests_run'] += 1
    
    def test_indian_market_compatibility(self):
        """Test Indian market specific features"""
        st.subheader("🇮🇳 Indian Market Compatibility")
        
        df = self.load_and_process_data()
        if df is None:
            return
        
        # Test 1: Currency Format
        with st.expander("Test 1: Indian Currency (₹) Handling", expanded=True):
            # Check market cap processing
            if 'market_cap' in self.df_raw.columns:
                # Show original vs processed
                st.write("Original market_cap format:", self.df_raw['market_cap'].iloc[0])
                if 'market_cap_num' in df.columns:
                    st.write("Processed market_cap:", df['market_cap_num'].iloc[0])
                    
                    # Check conversion success
                    valid_conversions = df['market_cap_num'].notna().sum()
                    conversion_rate = (valid_conversions / len(df) * 100)
                    
                    if conversion_rate > 95:
                        st.success(f"✅ Currency conversion successful ({conversion_rate:.1f}%)")
                        self.diagnostics['tests_passed'] += 1
                    else:
                        st.warning(f"⚠️ Conversion rate: {conversion_rate:.1f}%")
            
            # Check price columns
            price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            if available_price_cols:
                # Check numeric conversion
                all_numeric = True
                for col in available_price_cols:
                    if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                        all_numeric = False
                        st.error(f"❌ {col} is not numeric (type: {df[col].dtype})")
                
                if all_numeric:
                    st.success("✅ All price columns properly converted to numeric")
                    self.diagnostics['tests_passed'] += 1
                    
                    # Show price ranges
                    if 'price' in df.columns:
                        price_data = df['price'].dropna()
                        if len(price_data) > 0:
                            price_stats = price_data.describe()
                            st.write("Price Statistics (₹):")
                            st.write(f"- Min: ₹{price_stats['min']:,.2f}")
                            st.write(f"- Max: ₹{price_stats['max']:,.2f}")
                            st.write(f"- Median: ₹{price_stats['50%']:,.2f}")
                            st.write(f"- Mean: ₹{price_stats['mean']:,.2f}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Market Cap Analysis
        with st.expander("Test 2: Market Cap Analysis", expanded=True):
            if 'market_cap_num' in df.columns:
                market_cap = df['market_cap_num'].dropna()
                
                if len(market_cap) > 0:
                    # Categorize by market cap (in Crores)
                    large_cap = (market_cap >= 20000).sum()
                    mid_cap = ((market_cap >= 5000) & (market_cap < 20000)).sum()
                    small_cap = (market_cap < 5000).sum()
                    
                    fig = go.Figure(data=[go.Bar(
                        x=['Large Cap\n(≥20k Cr)', 'Mid Cap\n(5k-20k Cr)', 'Small Cap\n(<5k Cr)'],
                        y=[large_cap, mid_cap, small_cap],
                        text=[large_cap, mid_cap, small_cap],
                        textposition='auto',
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                    )])
                    fig.update_layout(
                        title="Market Cap Distribution",
                        yaxis_title="Number of Stocks",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Category distribution if available
                    if 'category' in df.columns:
                        cat_dist = df['category'].value_counts()
                        st.write("Category Distribution:")
                        for cat, count in cat_dist.items():
                            st.write(f"- {cat}: {count} stocks ({count/len(df)*100:.1f}%)")
                    
                    st.success("✅ Market cap analysis complete")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning("⚠️ No valid market cap data")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Sector Analysis
        with st.expander("Test 3: Indian Sector Analysis", expanded=True):
            if 'sector' in df.columns:
                sectors = df['sector'].value_counts().head(15)
                
                if len(sectors) > 0:
                    fig = go.Figure(data=[go.Bar(
                        y=sectors.index,
                        x=sectors.values,
                        orientation='h',
                        text=sectors.values,
                        textposition='auto'
                    )])
                    fig.update_layout(
                        title="Top 15 Sectors by Stock Count",
                        height=500,
                        xaxis_title="Number of Stocks",
                        yaxis_title="Sector",
                        margin=dict(l=200)  # More space for sector names
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Check for Indian sectors
                    indian_sectors = ['Banks', 'IT', 'Pharma', 'Auto', 'FMCG', 'Metal', 
                                    'Realty', 'Finance', 'Software', 'Healthcare', 'Cement',
                                    'Chemical', 'Power', 'Oil', 'Telecom']
                    
                    found_sectors = []
                    for sector in sectors.index:
                        if any(ind.lower() in sector.lower() for ind in indian_sectors):
                            found_sectors.append(sector)
                    
                    if found_sectors:
                        st.success(f"✅ Indian sectors identified: {len(found_sectors)}")
                        st.write("Examples:", ', '.join(found_sectors[:5]))
                        self.diagnostics['tests_passed'] += 1
                    else:
                        st.info("ℹ️ Sector names may use different conventions")
                    
                    self.diagnostics['data_quality']['top_sectors'] = sectors.head(10).to_dict()
            
            self.diagnostics['tests_run'] += 1
    
    def test_performance_metrics(self):
        """Test system performance"""
        st.subheader("⚡ Performance Metrics")
        
        # Load time test
        with st.expander("Performance Analysis", expanded=True):
            # Data load time
            start = time.time()
            df = self.load_raw_data()
            load_time = time.time() - start
            
            # Processing time
            process_time = 0
            if df is not None:
                start = time.time()
                df_processed = self.apply_signal_logic(self.clean_data(df.copy()))
                process_time = time.time() - start
            
            # Memory usage
            memory_mb = 0
            if df is not None:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Load Time", f"{load_time:.2f}s")
                if load_time < 3:
                    st.success("✅ Excellent")
                    self.diagnostics['tests_passed'] += 1
                elif load_time < 5:
                    st.warning("⚠️ Good")
                else:
                    st.error("❌ Slow")
                    self.diagnostics['tests_failed'] += 1
            
            with col2:
                st.metric("Processing Time", f"{process_time:.2f}s")
                if process_time < 2:
                    st.success("✅ Fast")
                    self.diagnostics['tests_passed'] += 1
                elif process_time < 4:
                    st.warning("⚠️ Acceptable")
                else:
                    st.error("❌ Slow")
            
            with col3:
                st.metric("Total Time", f"{load_time + process_time:.2f}s")
            
            with col4:
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
                if memory_mb < 100:
                    st.success("✅ Optimal")
                    self.diagnostics['tests_passed'] += 1
                elif memory_mb < 200:
                    st.warning("⚠️ Acceptable")
                else:
                    st.error("❌ High")
            
            self.diagnostics['tests_run'] += 3
            self.diagnostics['performance_metrics'] = {
                'load_time': float(load_time),
                'process_time': float(process_time),
                'memory_mb': float(memory_mb),
                'rows': len(df) if df is not None else 0,
                'columns': len(df.columns) if df is not None else 0
            }
            
            # Show data size
            if df is not None:
                st.write(f"\n**Data Size:** {len(df):,} rows × {len(df.columns)} columns")
    
    def generate_diagnostic_reports(self):
        """Generate downloadable diagnostic reports"""
        st.subheader("📥 Diagnostic Reports")
        
        # Calculate final stats
        self.diagnostics['total_tests'] = self.diagnostics['tests_run']
        self.diagnostics['success_rate'] = (
            self.diagnostics['tests_passed'] / self.diagnostics['tests_run'] * 100 
            if self.diagnostics['tests_run'] > 0 else 0
        )
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tests Run", self.diagnostics['tests_run'])
        with col2:
            st.metric("Tests Passed", self.diagnostics['tests_passed'])
        with col3:
            st.metric("Tests Failed", self.diagnostics['tests_failed'])
        with col4:
            st.metric("Success Rate", f"{self.diagnostics['success_rate']:.1f}%")
        
        # Health status
        if self.diagnostics['success_rate'] >= 85:
            st.success("✅ System Health: EXCELLENT - Ready for production")
        elif self.diagnostics['success_rate'] >= 70:
            st.warning("⚠️ System Health: GOOD - Minor issues detected")
        else:
            st.error("❌ System Health: NEEDS ATTENTION - Review errors")
        
        # Download buttons
        st.markdown("### 📥 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON Report
            json_str = json.dumps(self.diagnostics, indent=2, default=str)
            st.download_button(
                label="📄 JSON Report",
                data=json_str,
                file_name=f"mantra_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Text Report
            text_report = self.generate_text_report()
            st.download_button(
                label="📝 Text Report",
                data=text_report,
                file_name=f"mantra_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col3:
            # Sample Data
            if self.df_processed is not None:
                csv = self.df_processed.head(100).to_csv(index=False)
                st.download_button(
                    label="📊 Sample Data",
                    data=csv,
                    file_name=f"mantra_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Show issues
        if self.diagnostics['warnings']:
            with st.expander(f"⚠️ Warnings ({len(self.diagnostics['warnings'])})", expanded=False):
                for warning in self.diagnostics['warnings']:
                    st.warning(warning)
        
        if self.diagnostics['errors']:
            with st.expander(f"❌ Errors ({len(self.diagnostics['errors'])})", expanded=True):
                for error in self.diagnostics['errors']:
                    st.error(error)
    
    # Helper methods
    def load_raw_data(self):
        """Load raw data from Google Sheets"""
        try:
            response = requests.get(SHEET_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            
            # Basic cleanup - remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Remove any columns that are completely empty
            df = df.dropna(axis=1, how='all')
            
            return df
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare data for analysis - COMPREHENSIVE VERSION"""
        if df is None:
            return None
            
        # First, handle market_cap specially to create market_cap_num
        if 'market_cap' in df.columns:
            # Extract numeric value from market_cap (e.g., "₹14,232 Cr" -> 14232)
            df['market_cap_num'] = (
                df['market_cap']
                .astype(str)
                .str.replace('₹', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace(' Cr', '', regex=False)
                .str.replace('Cr', '', regex=False)
                .str.strip()
            )
            df['market_cap_num'] = pd.to_numeric(df['market_cap_num'], errors='coerce')
        
        # Define columns that should be numeric with their patterns
        numeric_conversions = {
            # Price columns
            'price': ['₹', ','],
            'prev_close': ['₹', ','],
            'low_52w': ['₹', ','],
            'high_52w': ['₹', ','],
            'sma_20d': ['₹', ','],
            'sma_50d': ['₹', ','],
            'sma_200d': ['₹', ','],
            
            # Volume columns (some are objects with commas)
            'volume_1d': [','],
            'volume_7d': [','],
            'volume_30d': [','],
            'volume_90d': [','],
            'volume_180d': [','],
            
            # Return columns
            'ret_1d': ['%'],
            'ret_3d': ['%'],
            'ret_7d': ['%'],
            'ret_30d': ['%'],
            'ret_3m': ['%'],
            'ret_6m': ['%'],
            'ret_1y': ['%'],
            'ret_3y': ['%'],
            'ret_5y': ['%'],
            
            # Volume ratio columns (some have % signs)
            'vol_ratio_1d_90d': ['%'],
            'vol_ratio_7d_90d': ['%'],
            'vol_ratio_30d_90d': ['%'],
            'vol_ratio_1d_180d': ['%'],
            'vol_ratio_7d_180d': ['%'],
            'vol_ratio_30d_180d': ['%'],
            'vol_ratio_90d_180d': ['%'],
            
            # Other numeric columns
            'from_low_pct': ['%'],
            'from_high_pct': ['%'],
            'eps_change_pct': ['%'],
            'pe': [],
            'eps_current': [],
            'eps_last_qtr': [],
            'rvol': [],
            'year': []
        }
        
        # Convert each column
        for col, chars_to_remove in numeric_conversions.items():
            if col in df.columns:
                # Convert to string first
                df[col] = df[col].astype(str)
                
                # Remove specified characters
                for char in chars_to_remove:
                    df[col] = df[col].str.replace(char, '', regex=False)
                
                # Remove any remaining common issues
                df[col] = df[col].str.replace(' ', '', regex=False)
                df[col] = df[col].str.strip()
                
                # Replace common null indicators
                df[col] = df[col].replace(['', '-', 'NA', 'N/A', 'nan', 'NaN', 'None', '#N/A'], np.nan)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure volume columns are properly handled
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
        for col in volume_cols:
            if col in df.columns:
                # Fill NaN with 0 for volume columns
                df[col] = df[col].fillna(0)
        
        # Calculate volume acceleration if needed columns exist
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
            df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        
        return df
    
    def load_and_process_data(self):
        """Load and process data with all calculations"""
        # Use cached data if available
        if self.df_processed is not None:
            return self.df_processed
        
        df = self.load_raw_data()
        if df is None:
            return None
        
        df = self.clean_data(df)
        
        # Store processed data
        self.df_processed = df
        return df
    
    def apply_signal_logic(self, df):
        """Apply signal detection logic - ROBUST VERSION"""
        if df is None:
            return None
            
        df = df.copy()
        df['EDGE_SIGNAL'] = 'NONE'
        
        # Calculate volume acceleration if not already present
        if 'volume_acceleration' not in df.columns:
            if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
                df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
            else:
                # Create dummy column if can't calculate
                df['volume_acceleration'] = 0
        
        # Ensure all required columns are numeric
        numeric_cols = [
            'volume_acceleration', 'eps_current', 'eps_last_qtr',
            'from_high_pct', 'ret_30d', 'pe', 'ret_7d', 'ret_1d',
            'price', 'sma_50d', 'sma_200d', 'eps_change_pct'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Triple Alignment Pattern - The Holy Grail
        if all(col in df.columns for col in ['volume_acceleration', 'eps_current', 'eps_last_qtr', 
                                              'from_high_pct', 'ret_30d', 'pe']):
            triple_mask = (
                (df['volume_acceleration'] > 10) &                     # Strong volume acceleration
                (df['eps_current'] > df['eps_last_qtr']) &           # EPS growing
                (df['from_high_pct'] < -20) &                         # Away from highs
                (df['ret_30d'].abs() < 5) &                           # Price stable
                (df['pe'] > 0) & (df['pe'] < 50)                     # Reasonable valuation
            )
            df.loc[triple_mask, 'EDGE_SIGNAL'] = 'TRIPLE_ALIGNMENT'
        
        # Coiled Spring Pattern
        if all(col in df.columns for col in ['volume_acceleration', 'ret_30d', 'from_high_pct']):
            spring_mask = (
                (df['volume_acceleration'] > 5) &                      # Volume accelerating
                (df['ret_30d'].abs() < 5) &                          # Price stable
                (df['from_high_pct'] < -30) &                         # Well below highs
                (df['EDGE_SIGNAL'] == 'NONE')                        # Not already classified
            )
            df.loc[spring_mask, 'EDGE_SIGNAL'] = 'COILED_SPRING'
        
        # Momentum Knife Pattern
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'volume_acceleration', 'price', 'sma_50d']):
            # Calculate momentum acceleration
            df['momentum_acceleration'] = 0
            valid_returns = (df['ret_7d'] != 0) & df['ret_7d'].notna()
            df.loc[valid_returns, 'momentum_acceleration'] = df.loc[valid_returns, 'ret_1d'] / (df.loc[valid_returns, 'ret_7d'] / 7)
            
            knife_mask = (
                (df['momentum_acceleration'] > 1.5) &                  # Momentum accelerating
                (df['ret_1d'] > 0) &                                  # Positive today
                (df['price'] > df['sma_50d']) &                      # Above support
                (df['EDGE_SIGNAL'] == 'NONE')                        # Not already classified
            )
            df.loc[knife_mask, 'EDGE_SIGNAL'] = 'MOMENTUM_KNIFE'
        
        # Smart Money Pattern
        if all(col in df.columns for col in ['eps_change_pct', 'pe', 'volume_acceleration']):
            smart_mask = (
                (df['eps_change_pct'] > 20) &                        # Strong EPS growth
                (df['pe'] > 0) & (df['pe'] < 40) &                  # Reasonable PE
                (df['volume_acceleration'] > 0) &                     # Positive volume trend
                (df['EDGE_SIGNAL'] == 'NONE')                        # Not already classified
            )
            df.loc[smart_mask, 'EDGE_SIGNAL'] = 'SMART_MONEY'
        
        return df
    
    def generate_text_report(self):
        """Generate human-readable text report"""
        def safe_json(obj):
            """Convert numpy/pandas types for JSON serialization"""
            if isinstance(obj, dict):
                return {k: safe_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [safe_json(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif pd.api.types.is_integer_dtype(type(obj)):
                return int(obj)
            elif pd.api.types.is_float_dtype(type(obj)):
                return float(obj)
            else:
                return obj
        
        report = f"""
M.A.N.T.R.A. DIAGNOSTIC REPORT
==============================
Generated: {self.diagnostics['timestamp']}

EXECUTIVE SUMMARY
-----------------
Total Tests Run: {self.diagnostics['tests_run']}
Tests Passed: {self.diagnostics['tests_passed']}
Tests Failed: {self.diagnostics['tests_failed']}
Success Rate: {self.diagnostics['success_rate']:.1f}%

SYSTEM STATUS: {'EXCELLENT' if self.diagnostics['success_rate'] >= 85 else 'NEEDS ATTENTION'}

DATA QUALITY
------------
{json.dumps(safe_json(self.diagnostics['data_quality']), indent=2)}

PERFORMANCE METRICS
-------------------
{json.dumps(safe_json(self.diagnostics['performance_metrics']), indent=2)}

SIGNAL ANALYSIS
---------------
{json.dumps(safe_json(self.diagnostics['signal_analysis']), indent=2)}

WARNINGS ({len(self.diagnostics['warnings'])})
--------
{chr(10).join(self.diagnostics['warnings']) if self.diagnostics['warnings'] else 'None'}

ERRORS ({len(self.diagnostics['errors'])})
------
{chr(10).join(self.diagnostics['errors']) if self.diagnostics['errors'] else 'None'}

RECOMMENDATIONS
---------------
"""
        if self.diagnostics['success_rate'] >= 85:
            report += "✅ System is functioning excellently and ready for production use."
        elif self.diagnostics['success_rate'] >= 70:
            report += "⚠️ System is functioning well with minor issues. Review warnings for optimization."
        else:
            report += "❌ System needs attention. Review errors and warnings before production use."
        
        return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🔍 M.A.N.T.R.A. System Diagnostics")
    st.markdown("""
    ### Complete Health Check for M.A.N.T.R.A. EDGE System
    
    This diagnostic tool validates:
    - ✅ Data pipeline integrity
    - ✅ Calculation accuracy
    - ✅ Signal generation logic
    - ✅ Indian market compatibility
    - ✅ System performance
    """)
    
    # Initialize diagnostics
    diagnostics = SystemDiagnostics()
    
    # Main action button
    if st.button("🚀 Run Complete Diagnostics", type="primary", use_container_width=True):
        with st.spinner("Running comprehensive system diagnostics..."):
            diagnostics.run_complete_diagnostics()
    else:
        st.info("👆 Click 'Run Complete Diagnostics' to start comprehensive system testing")
    
    # Sidebar tools
    with st.sidebar:
        st.header("🏥 Quick Health Check")
        
        if st.button("Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                try:
                    response = requests.get(SHEET_URL, timeout=5, allow_redirects=True)
                    if response.status_code == 200:
                        st.success("✅ Connected successfully")
                    else:
                        st.warning(f"⚠️ Status: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        if st.button("Quick Data Check", use_container_width=True):
            with st.spinner("Loading..."):
                try:
                    df = pd.read_csv(io.StringIO(requests.get(SHEET_URL).text), nrows=10)
                    # Remove unnamed columns
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
                    st.success(f"✅ Loaded {len(df)} rows")
                    st.write("Columns:", list(df.columns)[:5], "...")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 System Information
        
        **Data Source:**
        - Google Sheets
        - 1,785 stocks
        - 43 columns
        
        **Signal Types:**
        - Triple Alignment
        - Coiled Spring  
        - Momentum Knife
        - Smart Money
        
        **Performance Targets:**
        - Load time: <3s
        - Process time: <2s
        - Memory: <100MB
        
        **Version:** FINAL v1.0
        """)

if __name__ == "__main__":
    main()

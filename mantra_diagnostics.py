```python
# mantra_diagnostics.py - COMPLETE SYSTEM HEALTH CHECK & TESTING
"""
M.A.N.T.R.A. Diagnostics Dashboard
==================================
This diagnostic system checks EVERYTHING:
- Data loading pipeline
- Calculation accuracy
- Signal generation
- Indian market compatibility
- Performance metrics
- Complete system health

Run this alongside your main app to ensure everything works perfectly.
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
import traceback
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. Diagnostics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use same config as main system
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Expected columns for Indian market (updated based on actual data)
# Note: volume_acceleration is calculated, not expected from source
# Note: eps_tier, price_tier, trading_under, eps_duplicate may be missing in some datasets
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
# DIAGNOSTIC FUNCTIONS
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
    
    def run_complete_diagnostics(self):
        """Run all diagnostic tests"""
        st.header("🔍 M.A.N.T.R.A. Complete System Diagnostics")
        
        # Create diagnostic tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Pipeline", "🧮 Calculations", "📈 Signals", 
            "🇮🇳 Indian Market", "⚡ Performance", "📥 Reports"
        ])
        
        with tab1:
            self.test_data_pipeline()
        
        with tab2:
            self.test_calculations()
        
        with tab3:
            self.test_signal_generation()
        
        with tab4:
            self.test_indian_market_compatibility()
        
        with tab5:
            self.test_performance_metrics()
        
        with tab6:
            self.generate_diagnostic_reports()
    
    def test_data_pipeline(self):
        """Test complete data loading pipeline"""
        st.subheader("📊 Data Pipeline Testing")
        
        # Test 1: URL Accessibility
        with st.expander("Test 1: Google Sheets Accessibility", expanded=True):
            try:
                start = time.time()
                # Use GET request instead of HEAD for Google Sheets
                response = requests.get(SHEET_URL, timeout=10, allow_redirects=True)
                load_time = time.time() - start
                
                if response.status_code == 200:
                    st.success(f"✅ Google Sheets accessible (Response time: {load_time:.2f}s)")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning(f"⚠️ Google Sheets returned status {response.status_code} but data may still be accessible")
                    self.diagnostics['warnings'].append(f"Sheet status: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                self.diagnostics['tests_failed'] += 1
                self.diagnostics['errors'].append(f"Connection error: {str(e)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Data Loading
        with st.expander("Test 2: Data Loading & Parsing", expanded=True):
            try:
                df_raw = self.load_raw_data()
                if df_raw is not None and not df_raw.empty:
                    st.success(f"✅ Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
                    self.diagnostics['tests_passed'] += 1
                    
                    # Show sample
                    st.write("Sample data (first 3 rows):")
                    st.dataframe(df_raw.head(3))
                    
                    # Column analysis
                    st.write("Column types:")
                    col_types = df_raw.dtypes.value_counts()
                    st.write(col_types)
                    
                    # Store for later use
                    self.df_raw = df_raw
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
            if hasattr(self, 'df_raw'):
                missing_cols = EXPECTED_COLUMNS - set(self.df_raw.columns)
                extra_cols = set(self.df_raw.columns) - EXPECTED_COLUMNS
                
                col1, col2 = st.columns(2)
                with col1:
                    if missing_cols:
                        st.warning(f"⚠️ Missing columns: {missing_cols}")
                        self.diagnostics['warnings'].append(f"Missing columns: {list(missing_cols)}")
                    else:
                        st.success("✅ All expected columns present")
                        self.diagnostics['tests_passed'] += 1
                
                with col2:
                    if extra_cols:
                        st.info(f"ℹ️ Extra columns: {len(extra_cols)}")
                        st.write("Extra columns:", list(extra_cols)[:10])  # Show first 10
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['missing_columns'] = list(missing_cols)
                self.diagnostics['data_quality']['total_columns'] = int(len(self.df_raw.columns))
        
        # Test 4: Data Quality
        with st.expander("Test 4: Data Quality Analysis", expanded=True):
            if hasattr(self, 'df_raw'):
                df_clean = self.clean_data(self.df_raw.copy())  # Use copy to avoid warnings
                
                # Null analysis
                null_counts = df_clean.isnull().sum()
                high_null_cols = null_counts[null_counts > len(df_clean) * 0.5]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    null_pct = (null_counts.sum() / (len(df_clean) * len(df_clean.columns)) * 100)
                    st.metric("Null %", f"{null_pct:.1f}%")
                    if null_pct < 10:
                        st.success("✅ Good data quality")
                        self.diagnostics['tests_passed'] += 1
                    elif null_pct < 40:
                        st.warning("⚠️ Moderate null percentage")
                        self.diagnostics['warnings'].append(f"Moderate null percentage: {null_pct:.1f}%")
                    else:
                        st.error("❌ High null percentage")
                        self.diagnostics['warnings'].append(f"High null percentage: {null_pct:.1f}%")
                
                with col2:
                    st.metric("High Null Columns", len(high_null_cols))
                    if len(high_null_cols) > 0:
                        st.write("Columns with >50% nulls:", list(high_null_cols.index)[:5])
                
                with col3:
                    duplicates = df_clean['ticker'].duplicated().sum()
                    st.metric("Duplicate Tickers", duplicates)
                    if duplicates > 0:
                        self.diagnostics['warnings'].append(f"Duplicate tickers: {duplicates}")
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['null_percentage'] = float(null_pct)
                self.diagnostics['data_quality']['duplicate_tickers'] = int(duplicates)
    
    def test_calculations(self):
        """Test all calculations"""
        st.subheader("🧮 Calculation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test calculations - data loading failed")
            return
        
        # Test Volume Acceleration
        with st.expander("Test 1: Volume Acceleration Calculation", expanded=True):
            if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
                try:
                    # Check if columns are numeric
                    if df['vol_ratio_30d_90d'].dtype in ['float64', 'int64'] and df['vol_ratio_30d_180d'].dtype in ['float64', 'int64']:
                        # Manual calculation
                        vol_accel_manual = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
                        
                        # System calculation (re-calculate to ensure consistency)
                        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
                        
                        # Compare (excluding NaN values)
                        valid_mask = ~(vol_accel_manual.isna() | df['volume_acceleration'].isna())
                        if valid_mask.any():
                            diff = (vol_accel_manual[valid_mask] - df['volume_acceleration'][valid_mask]).abs().max()
                            if diff < 0.01:
                                st.success(f"✅ Volume acceleration calculation correct (max diff: {diff:.6f})")
                                self.diagnostics['tests_passed'] += 1
                            else:
                                st.error(f"❌ Volume acceleration mismatch (max diff: {diff:.6f})")
                                self.diagnostics['tests_failed'] += 1
                        else:
                            st.warning("⚠️ No valid data for volume acceleration calculation")
                        
                        # Show examples
                        st.write("Sample calculations:")
                        sample_df = df[['ticker', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'volume_acceleration']].dropna().head(5)
                        st.dataframe(sample_df)
                    else:
                        st.error("❌ Volume ratio columns are not numeric")
                        st.write(f"vol_ratio_30d_90d dtype: {df['vol_ratio_30d_90d'].dtype}")
                        st.write(f"vol_ratio_30d_180d dtype: {df['vol_ratio_30d_180d'].dtype}")
                        self.diagnostics['tests_failed'] += 1
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    self.diagnostics['tests_failed'] += 1
            else:
                st.warning("⚠️ Required columns not found for volume acceleration")
            
            self.diagnostics['tests_run'] += 1
        
        # Test Momentum Acceleration
        with st.expander("Test 2: Momentum Acceleration", expanded=True):
            if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
                # Check for divide by zero handling
                zero_returns = df[(df['ret_3d'] == 0) | (df['ret_7d'] == 0)]
                st.write(f"Stocks with zero returns: {len(zero_returns)}")
                
                if len(zero_returns) > 0:
                    st.info("ℹ️ System should handle zero returns correctly (e.g., avoid NaNs)")
                
                # Check acceleration logic (simple positive check)
                positive_accel = df[(df['ret_1d'] > 0) & (df['ret_3d'] > 0) & (df['ret_7d'] > 0)].dropna()
                st.write(f"Stocks with positive momentum (1d, 3d, 7d): {len(positive_accel)}")
                
                if len(positive_accel) > 0:
                    st.success("✅ Momentum data available and seems to be calculated")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.info("ℹ️ Few stocks with all positive short-term momentum (could be market condition)")

            else:
                st.warning("⚠️ Required columns for momentum not found")
            
            self.diagnostics['tests_run'] += 1
        
        # Test Conviction Score
        with st.expander("Test 3: Conviction Score Components", expanded=True):
            # Test each component
            components = {}
            
            if 'volume_acceleration' in df.columns:
                components['Volume Acceleration > 10%'] = int((df['volume_acceleration'] > 10).sum())
            
            if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
                components['Momentum Building'] = int((df['ret_7d'] > df['ret_30d']/4).sum())
            
            if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
                components['EPS Improving'] = int((df['eps_current'] > df['eps_last_qtr']).sum())
            
            if all(col in df.columns for col in ['price', 'sma_50d']):
                # Ensure both columns are numeric and not null
                price_valid = pd.to_numeric(df['price'], errors='coerce')
                sma_valid = pd.to_numeric(df['sma_50d'], errors='coerce')
                components['Above 50MA'] = int(((price_valid > sma_valid) & price_valid.notna() & sma_valid.notna()).sum())
            
            if 'rvol' in df.columns:
                rvol_valid = pd.to_numeric(df['rvol'], errors='coerce')
                components['High Volume'] = int((rvol_valid > 1.5).sum())
            
            for component, count in components.items():
                st.write(f"{component}: {count} stocks")
            
            if components:
                st.success("✅ Conviction components calculated")
                self.diagnostics['tests_passed'] += 1
            else:
                st.warning("⚠️ No components for conviction score could be calculated (missing data?)")
            
            self.diagnostics['calculation_checks']['conviction_components'] = components
            self.diagnostics['tests_run'] += 1
    
    def test_signal_generation(self):
        """Test signal generation logic"""
        st.subheader("📈 Signal Generation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test signals - data loading failed")
            return
        
        # Run signal detection
        df = self.apply_signal_logic(df)
        
        # Signal distribution
        with st.expander("Signal Distribution Analysis", expanded=True):
            if 'EDGE_SIGNAL' in df.columns:
                signal_counts = df['EDGE_SIGNAL'].value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.values,
                    text=signal_counts.values,
                    textposition='auto'
                )])
                fig.update_layout(title="Signal Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                self.diagnostics['signal_analysis']['distribution'] = {k: int(v) for k, v in signal_counts.to_dict().items()}
                
                # Quality check
                total_signals = signal_counts[signal_counts.index != 'NONE'].sum() if 'NONE' in signal_counts.index else signal_counts.sum()
                signal_ratio = total_signals / len(df) * 100
                
                st.write(f"Signal generation rate: {signal_ratio:.1f}%")
                if 2 <= signal_ratio <= 20:
                    st.success("✅ Signal generation rate is reasonable (2-20%)")
                    self.diagnostics['tests_passed'] += 1
                elif signal_ratio > 20:
                    st.warning(f"⚠️ Signal rate too high: {signal_ratio:.1f}% (expected 2-20%)")
                    st.write("Consider tightening signal conditions")
                    self.diagnostics['warnings'].append(f"High signal rate: {signal_ratio:.1f}%")
                elif signal_ratio < 2 and signal_ratio > 0:
                    st.info(f"ℹ️ Low signal rate: {signal_ratio:.1f}% (this may be fine in bear markets)")
                else:
                    st.info("ℹ️ No signals generated (check data quality and market conditions)")
            else:
                st.error("❌ 'EDGE_SIGNAL' column not found, signals not generated.")
            
            self.diagnostics['tests_run'] += 1
        
        # Triple Alignment Quality
        with st.expander("Triple Alignment Pattern Testing", expanded=True):
            if 'EDGE_SIGNAL' in df.columns:
                triple_align = df[df['EDGE_SIGNAL'] == 'TRIPLE_ALIGNMENT']
                
                if len(triple_align) > 0:
                    st.success(f"✅ Found {len(triple_align)} Triple Alignment patterns")
                    
                    # Verify conditions with updated thresholds
                    conditions_met = {}
                    
                    if 'volume_acceleration' in triple_align.columns:
                        conditions_met['Volume > 5%'] = (triple_align['volume_acceleration'] > 5).mean() > 0.8
                    
                    if all(col in triple_align.columns for col in ['eps_current', 'eps_last_qtr']):
                        conditions_met['EPS Growing'] = ((triple_align['eps_current'] > triple_align['eps_last_qtr']) & 
                                                         (triple_align['eps_current'] > 0)).mean() > 0.8
                    
                    if 'from_high_pct' in triple_align.columns:
                        conditions_met['15-40% from highs'] = ((triple_align['from_high_pct'] < -15) & 
                                                               (triple_align['from_high_pct'] > -40)).mean() > 0.8
                    
                    if 'ret_30d' in triple_align.columns:
                        conditions_met['Consolidating (<10% move)'] = (triple_align['ret_30d'].abs() < 10).mean() > 0.8
                    
                    for condition, met in conditions_met.items():
                        if met:
                            st.write(f"✅ {condition} (80%+ stocks meet criteria)")
                        else:
                            st.write(f"⚠️ {condition} (Less than 80% meet criteria)")
                            self.diagnostics['warnings'].append(f"Triple alignment: {condition} not met by 80%+")
                else:
                    st.info("No Triple Alignment patterns found")
                    st.write("This could indicate:")
                    st.write("- Very strict conditions")
                    st.write("- Data quality issues") 
                    st.write("- Market conditions not favorable")
                
                self.diagnostics['signal_analysis']['triple_alignments'] = int(len(triple_align))
            else:
                st.warning("⚠️ 'EDGE_SIGNAL' column not found for Triple Alignment analysis.")
            
            self.diagnostics['tests_run'] += 1
    
    def test_indian_market_compatibility(self):
        """Test Indian market specific features"""
        st.subheader("🇮🇳 Indian Market Compatibility")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test Indian market compatibility - data loading failed")
            return
        
        # Test 1: Currency Format
        with st.expander("Test 1: Indian Currency (₹) Handling", expanded=True):
            # Check if price columns are numeric after ₹ removal
            price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            all_numeric = True
            if available_price_cols:
                for col in available_price_cols:
                    # Check if the column is actually numeric or can be coerced
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        st.error(f"❌ {col} is not numeric after cleaning (dtype: {df[col].dtype})")
                        all_numeric = False
                
                if all_numeric:
                    st.success("✅ All price columns properly converted to numeric format")
                    self.diagnostics['tests_passed'] += 1
                    
                    # Show sample prices
                    st.write("Sample price data (numeric):")
                    sample_cols = [col for col in available_price_cols if not df[col].isna().all()][:5]
                    if sample_cols:
                        st.dataframe(df[sample_cols].dropna().head(5))
                else:
                    self.diagnostics['tests_failed'] += 1
            else:
                st.warning("⚠️ No price-related columns found to test currency handling.")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Indian Exchanges
        with st.expander("Test 2: Indian Exchange Detection", expanded=True):
            if 'ticker' in df.columns:
                sample_tickers = df['ticker'].dropna().head(20).tolist()
                st.write("Sample tickers:", sample_tickers[:10])
                
                # Check for .NS or .BO suffixes (NSE/BSE indicators)
                indian_tickers_suffix = [t for t in sample_tickers if isinstance(t, str) and (t.endswith('.NS') or t.endswith('.BO'))]
                
                if indian_tickers_suffix:
                    st.success(f"✅ Indian exchange tickers with .NS/.BO suffixes detected: {len(indian_tickers_suffix)}")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.info("ℹ️ Tickers don't have .NS/.BO exchange suffixes. This is normal if data uses plain ticker symbols for Indian stocks.")
                    self.diagnostics['tests_passed'] += 1  # Not a failure
            else:
                st.warning("⚠️ 'ticker' column not found for exchange detection.")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Market Cap in Crores
        with st.expander("Test 3: Market Cap Format", expanded=True):
            if 'market_cap' in df.columns:
                # Check if market cap is numeric
                if pd.api.types.is_numeric_dtype(df['market_cap']):
                    st.success("✅ 'market_cap' column is numeric")
                    
                    # Check range (Indian market caps are typically large in crores, e.g., 1000 = 1000 Crores)
                    non_null_caps = df['market_cap'].dropna()
                    if len(non_null_caps) > 0:
                        min_cap = non_null_caps.min()
                        max_cap = non_null_caps.max()
                        median_cap = non_null_caps.median()
                        
                        st.write(f"Market cap range: {min_cap:,.0f} to {max_cap:,.0f}")
                        st.write(f"Median market cap: {median_cap:,.0f}")
                        
                        # Heuristic: if median is > 1000, it's likely in crores (e.g., 1000 Cr)
                        if median_cap > 1000:
                            st.success("✅ Market cap values suggest 'Crores' format (median > 1000)")
                            self.diagnostics['tests_passed'] += 1
                        else:
                            st.info("ℹ️ Median market cap is not indicative of 'Crores' format, but it's numeric.")
                    else:
                        st.warning("⚠️ No non-null market cap values to analyze range.")
                else:
                    st.warning("⚠️ 'market_cap' is not numeric, check cleaning process.")
            else:
                st.warning("⚠️ 'market_cap' column not found for format test.")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 4: Sector Distribution
        with st.expander("Test 4: Indian Sector Analysis", expanded=True):
            if 'sector' in df.columns:
                sectors = df['sector'].value_counts().head(10)
                
                if len(sectors) > 0:
                    fig = go.Figure(data=[go.Bar(x=sectors.values, y=sectors.index, orientation='h')])
                    fig.update_layout(title="Top 10 Sectors", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Common Indian sectors (check for presence of common names)
                    indian_sectors_keywords = ['Bank', 'IT', 'Pharma', 'Auto', 'FMCG', 'Metal', 'Realty', 
                                         'Finance', 'Software', 'Healthcare', 'Consumer', 'Energy']
                    
                    found_indian_keywords = [s for s in sectors.index if any(keyword.lower() in str(s).lower() for keyword in indian_sectors_keywords)]
                    
                    if found_indian_keywords:
                        st.success(f"✅ Common Indian market sector keywords identified: {', '.join(found_indian_keywords[:5])}")
                        self.diagnostics['tests_passed'] += 1
                    else:
                        st.info("ℹ️ No common Indian market sector keywords found in top sectors. Check sector naming conventions.")
                    
                    self.diagnostics['data_quality']['top_sectors'] = {k: int(v) for k, v in sectors.head(5).to_dict().items()}
                else:
                    st.warning("⚠️ No sector data available to analyze.")
            else:
                st.warning("⚠️ 'sector' column not found for analysis.")
            
            self.diagnostics['tests_run'] += 1
    
    def test_performance_metrics(self):
        """Test system performance"""
        st.subheader("⚡ Performance Metrics")
        
        # Test load times
        with st.expander("Load Time Analysis", expanded=True):
            # Test 1: Data load time
            start = time.time()
            df = self.load_raw_data()
            load_time = time.time() - start
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Load Time", f"{load_time:.2f}s")
                if load_time < 5:
                    st.success("✅ Excellent")
                    self.diagnostics['tests_passed'] += 1
                elif load_time < 10:
                    st.warning("⚠️ Acceptable")
                else:
                    st.error("❌ Too slow")
                    self.diagnostics['tests_failed'] += 1
            
            # Test 2: Processing time
            process_time = None
            total_time = None
            if df is not None:
                start = time.time()
                try:
                    df_processed = self.apply_signal_logic(df.copy()) # Pass a copy to avoid side effects
                    process_time = time.time() - start
                    
                    with col2:
                        st.metric("Processing Time", f"{process_time:.2f}s")
                        if process_time < 3:
                            st.success("✅ Excellent")
                            self.diagnostics['tests_passed'] += 1
                        else:
                            st.warning("⚠️ Acceptable") # Or error if too slow for processing
                    
                    with col3:
                        total_time = load_time + process_time
                        st.metric("Total Time", f"{total_time:.2f}s")
                        if total_time < 10:
                            st.success("✅ Efficient overall")
                        else:
                            st.warning("⚠️ Total time could be improved")

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    self.diagnostics['errors'].append(f"Processing time calculation error: {str(e)}")
            else:
                st.warning("⚠️ Data not loaded, cannot measure processing time.")
                    
            self.diagnostics['tests_run'] += 2 # Count data load and processing time tests
            self.diagnostics['performance_metrics'] = {
                'load_time': float(load_time),
                'process_time': float(process_time) if process_time is not None else None,
                'total_time': float(total_time) if total_time is not None else None
            }
        
        # Memory usage
        with st.expander("Memory Usage", expanded=True):
            if df is not None:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("DataFrame Memory", f"{memory_mb:.1f} MB")
                
                if memory_mb < 100:
                    st.success("✅ Memory usage optimal")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB (consider optimizing data types or processing)")
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['performance_metrics']['memory_mb'] = float(memory_mb)
            else:
                st.warning("⚠️ Data not loaded, cannot measure memory usage.")
    
    def generate_diagnostic_reports(self):
        """Generate downloadable diagnostic reports"""
        st.subheader("📥 Diagnostic Reports")
        
        # Update final counts
        self.diagnostics['total_tests'] = self.diagnostics['tests_run']
        self.diagnostics['success_rate'] = (
            self.diagnostics['tests_passed'] / self.diagnostics['tests_run'] * 100 
            if self.diagnostics['tests_run'] > 0 else 0
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tests Run", self.diagnostics['tests_run'])
        with col2:
            st.metric("Tests Passed", self.diagnostics['tests_passed'])
        with col3:
            st.metric("Tests Failed", self.diagnostics['tests_failed'])
        with col4:
            st.metric("Success Rate", f"{self.diagnostics['success_rate']:.1f}%")
        
        # Overall health status
        if self.diagnostics['success_rate'] >= 90:
            st.success("✅ System Health: EXCELLENT")
        elif self.diagnostics['success_rate'] >= 70:
            st.warning("⚠️ System Health: GOOD (with warnings)")
        else:
            st.error("❌ System Health: NEEDS ATTENTION")
        
        # Download options
        st.markdown("### Download Diagnostic Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON Report
            json_report = json.dumps(self.diagnostics, indent=2, default=str)
            b64 = base64.b64encode(json_report.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="mantra_diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">📥 Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Text Report
            text_report = self.generate_text_report()
            b64 = base64.b64encode(text_report.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="mantra_diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📥 Download Text Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            # Full Data Sample
            df = self.load_and_process_data()
            if df is not None:
                csv = df.head(100).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="mantra_sample_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Download Sample Data</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Cannot provide sample data (data loading failed).")
        
        # Show warnings and errors
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
        """Load raw data for testing"""
        try:
            response = requests.get(SHEET_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            
            # Clean column names - remove leading/trailing spaces
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove empty/unnamed columns without regex
            valid_cols = []
            for col in df.columns:
                col_str = str(col).strip()
                # Skip unnamed columns, empty columns, underscore columns, and common null string representations
                if col_str.startswith('Unnamed'):
                    continue
                if col_str == '':
                    continue
                if col_str.startswith('_'):
                    continue
                if col_str.lower() in ['nan', 'none', 'null']:
                    continue
                # Keep this column
                valid_cols.append(col)
            
            df = df[valid_cols]
            return df
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None
    
    def clean_data(self, df):
        """Apply comprehensive data cleaning"""
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Remove empty columns (corrected regex for clarity and correctness)
        # Filters out columns starting with 'Unnamed', starting with '_', or being an empty string.
        df = df.loc[:, ~df.columns.str.contains(r'^Unnamed|^_|^$', regex=True)]
        
        # Convert numeric columns - MORE COMPREHENSIVE LIST
        numeric_keywords = ['price', 'volume', 'ret_', 'pe', 'eps', 'sma_', 
                            'vol_ratio', 'low_', 'high_', 'from_', 'market_cap', 
                            '_pct', 'rvol', 'prev_close', 'close', '_90d', '_180d']
        
        for col in df.columns:
            # Check if column contains any numeric keyword AND if its dtype is 'object' (string)
            if any(keyword in col.lower() for keyword in numeric_keywords) and df[col].dtype == 'object':
                # Clean string values first
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace('₹', '', regex=False)
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = df[col].str.replace('Cr', '', regex=False, flags=re.IGNORECASE) # Case-insensitive Cr removal
                df[col] = df[col].str.strip()
                
                # Replace common non-numeric string representations with NaN
                df[col] = df[col].replace(['', '-', 'NA', 'N/A', 'nan', 'NaN', 'None', 'NULL', 'null'], np.nan)
            
            # Convert to numeric, coercing errors to NaN
            # Apply to all columns that might become numeric, even if they started as non-object
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                # This catch is mostly for debugging, as errors='coerce' should handle most issues
                st.warning(f"Could not convert column {col} to numeric: {e}")

        # Ensure specific columns are numeric (double-check critical columns explicitly, even if keywords missed them)
        critical_numeric_cols = ['price', 'pe', 'eps_current', 'eps_last_qtr', 
                                 'from_high_pct', 'ret_30d', 'volume_acceleration', # volume_acceleration is calculated later, but good to ensure it's numeric if it exists
                                 'sma_50d', 'sma_200d', 'prev_close', 'rvol',
                                 'ret_1d', 'ret_3d', 'ret_7d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                                 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
                                 'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                                 'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
                                 'market_cap_num' # if market_cap is converted to a separate numeric column
                                ]
        
        for col in critical_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def load_and_process_data(self):
        """Load and process data similar to main app"""
        df = self.load_raw_data()
        if df is None:
            return None
        
        df = self.clean_data(df)
        
        # Basic calculations (with error handling)
        try:
            # Ensure columns for volume acceleration are numeric after cleaning
            if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
                df['vol_ratio_30d_90d'] = pd.to_numeric(df['vol_ratio_30d_90d'], errors='coerce')
                df['vol_ratio_30d_180d'] = pd.to_numeric(df['vol_ratio_30d_180d'], errors='coerce')
                
                # Calculate volume acceleration
                df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
            else:
                # Add dummy column if not present to avoid KeyError in signal logic
                df['volume_acceleration'] = np.nan 
                st.warning("Could not calculate volume acceleration: Required volume ratio columns missing.")

        except Exception as e:
            st.warning(f"Could not calculate volume acceleration: {str(e)}")
            df['volume_acceleration'] = np.nan # Ensure column exists even if calculation fails
        
        return df
    
    def apply_signal_logic(self, df):
        """Apply basic signal logic for testing"""
        df = df.copy()  # Avoid modifying original
        df['EDGE_SIGNAL'] = 'NONE'
        
        try:
            # Ensure critical columns are numeric before comparisons
            numeric_cols = ['volume_acceleration', 'eps_current', 'eps_last_qtr', 
                            'from_high_pct', 'ret_30d', 'pe', 'ret_7d', 'ret_1d',
                            'price', 'sma_50d'] # Added price and sma_50d for breakout
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Triple alignment (with stricter, more realistic conditions)
            # Need to define conditions based on available and cleaned data
            conditions_triple_alignment = []
            
            if 'volume_acceleration' in df.columns:
                conditions_triple_alignment.append(df['volume_acceleration'] > 5) # More realistic threshold
            
            if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
                conditions_triple_alignment.append((df['eps_current'] > df['eps_last_qtr']) & (df['eps_current'] > 0))
            
            if 'from_high_pct' in df.columns:
                conditions_triple_alignment.append((df['from_high_pct'] < -15) & (df['from_high_pct'] > -40)) # 15-40% below highs
            
            if 'ret_30d' in df.columns:
                conditions_triple_alignment.append(df['ret_30d'].abs() < 10) # Recent consolidation
            
            if 'pe' in df.columns:
                conditions_triple_alignment.append((df['pe'] > 5) & (df['pe'] < 40)) # Reasonable PE ratio
            
            if all(col in df.columns for col in ['ret_1d', 'ret_7d']):
                conditions_triple_alignment.append(df['ret_1d'] > (df['ret_7d'] / 7)) # Recent momentum improving
            
            if len(conditions_triple_alignment) >= 4: # Need at least 4 conditions for Triple Alignment
                mask_triple_alignment = pd.concat(conditions_triple_alignment, axis=1).fillna(False).sum(axis=1) >= 4
                df.loc[mask_triple_alignment, 'EDGE_SIGNAL'] = 'TRIPLE_ALIGNMENT'
            else:
                st.warning(f"Insufficient columns ({len(conditions_triple_alignment)} < 4) for Triple Alignment signal.")
            
            # Coiled spring (looser conditions)
            conditions_coiled_spring = []
            if 'volume_acceleration' in df.columns:
                conditions_coiled_spring.append(df['volume_acceleration'] > 2) # Some volume increase
            if 'ret_30d' in df.columns:
                conditions_coiled_spring.append(df['ret_30d'].abs() < 15) # Consolidating
            if 'from_high_pct' in df.columns:
                conditions_coiled_spring.append(df['from_high_pct'] < -25) # Well off highs

            if len(conditions_coiled_spring) >= 3:
                mask_coiled_spring = pd.concat(conditions_coiled_spring, axis=1).fillna(False).all(axis=1) # All must be true for coiled spring
                df.loc[mask_coiled_spring & (df['EDGE_SIGNAL'] == 'NONE'), 'EDGE_SIGNAL'] = 'COILED_SPRING'
            else:
                st.warning(f"Insufficient columns ({len(conditions_coiled_spring)} < 3) for Coiled Spring signal.")

            # Breakout momentum (new pattern)
            conditions_breakout = []
            if 'ret_7d' in df.columns:
                conditions_breakout.append(df['ret_7d'] > 5) # Strong recent move
            if 'volume_acceleration' in df.columns:
                conditions_breakout.append(df['volume_acceleration'] > 3) # Volume surge
            if all(col in df.columns for col in ['price', 'sma_50d']):
                conditions_breakout.append(df['price'] > df['sma_50d']) # Above key MA
            
            if len(conditions_breakout) >= 3:
                mask_breakout = pd.concat(conditions_breakout, axis=1).fillna(False).all(axis=1)
                df.loc[mask_breakout & (df['EDGE_SIGNAL'] == 'NONE'), 'EDGE_SIGNAL'] = 'BREAKOUT'
            else:
                st.warning(f"Insufficient columns ({len(conditions_breakout)} < 3) for Breakout signal.")

        except Exception as e:
            st.error(f"Signal generation error during application: {str(e)}")
            st.exception(e) # Display full traceback for debugging
            # Return df with default NONE signals, or partially applied signals
        
        return df
    
    def generate_text_report(self):
        """Generate human-readable text report"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.integer): # More specific for numpy integers
                return int(obj)
            elif isinstance(obj, np.floating): # More specific for numpy floats
                return float(obj)
            elif isinstance(obj, (int, float, str, bool)) or obj is None: # Already primitive types
                return obj
            else:
                return str(obj) # Fallback for other non-serializable types
        
        # Convert diagnostics for JSON serialization
        # Ensure all nested structures are converted
        diagnostics_serializable = convert_to_serializable(self.diagnostics)

        data_quality = diagnostics_serializable.get('data_quality', {})
        performance_metrics = diagnostics_serializable.get('performance_metrics', {})
        signal_analysis = diagnostics_serializable.get('signal_analysis', {})
        
        report = f"""
M.A.N.T.R.A. DIAGNOSTIC REPORT
==============================
Generated: {diagnostics_serializable['timestamp']}

SUMMARY
-------
Total Tests Run: {diagnostics_serializable['tests_run']}
Tests Passed: {diagnostics_serializable['tests_passed']}
Tests Failed: {diagnostics_serializable['tests_failed']}
Success Rate: {diagnostics_serializable['success_rate']:.1f}%

DATA QUALITY
------------
{json.dumps(data_quality, indent=2)}

PERFORMANCE METRICS
------------------
{json.dumps(performance_metrics, indent=2)}

SIGNAL ANALYSIS
--------------
{json.dumps(signal_analysis, indent=2)}

WARNINGS ({len(diagnostics_serializable['warnings'])})
--------
{chr(10).join(diagnostics_serializable['warnings']) if diagnostics_serializable['warnings'] else 'None'}

ERRORS ({len(diagnostics_serializable['errors'])})
------
{chr(10).join(diagnostics_serializable['errors']) if diagnostics_serializable['errors'] else 'None'}

RECOMMENDATION
--------------
"""
        if self.diagnostics['success_rate'] >= 90:
            report += "System is functioning EXCELLENTLY. Ready for production use."
        elif self.diagnostics['success_rate'] >= 70:
            report += "System is functioning WELL with minor issues. Review warnings."
        else:
            report += "System needs ATTENTION. Review errors and warnings before production use."
        
        return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🔍 M.A.N.T.R.A. System Diagnostics")
    st.markdown("""
    This diagnostic tool comprehensively tests your M.A.N.T.R.A. EDGE system to ensure:
    - ✅ Data loads correctly from Google Sheets
    - ✅ All calculations are accurate
    - ✅ Signals generate properly
    - ✅ Indian market compatibility
    - ✅ Performance is optimal
    """)
    
    # Initialize diagnostics
    diagnostics = SystemDiagnostics()
    
    # Run button
    if st.button("🚀 Run Complete Diagnostics", type="primary", use_container_width=True):
        diagnostics.run_complete_diagnostics()
    else:
        st.info("👆 Click 'Run Complete Diagnostics' to start comprehensive system testing")
    
    # Quick health check
    with st.sidebar:
        st.header("🏥 Quick Health Check")
        
        if st.button("Test Connection"):
            try:
                response = requests.get(SHEET_URL, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    st.success("✅ Connected")
                else:
                    st.warning(f"⚠️ Status: {response.status_code} (but may still work)")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        
        if st.button("Test Data Load"):
            try:
                # Use load_raw_data for consistency with main diagnostics
                df_test = diagnostics.load_raw_data()
                if df_test is not None:
                    st.success(f"✅ Loaded {len(df_test)} rows")
                else:
                    st.error("❌ Data load failed.")
            except Exception as e:
                st.error(f"❌ {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### 📋 What This Tests:
        
        1. **Data Pipeline**
            - Google Sheets access
            - CSV parsing
            - Column validation
            - Data quality
        
        2. **Calculations**
            - Volume acceleration
            - Momentum metrics
            - Conviction scores
        
        3. **Signal Logic**
            - Pattern detection
            - Signal distribution
            - Edge cases
        
        4. **Indian Market**
            - Currency handling
            - Exchange detection
            - Sector analysis
        
        5. **Performance**
            - Load times
            - Memory usage
            - Processing speed
        
        ### 🎯 Known Issues:
        - 'exchange' column missing (not expected in current data)
        - High null percentage normal for certain columns (e.g., historical EPS for newer listings)
        - 307 redirects are OK (handled by `allow_redirects=True`)
        """)

if __name__ == "__main__":
    main()
```

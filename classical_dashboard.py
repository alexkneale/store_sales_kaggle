import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Ecuador Store Sales: Classical Diagnostics",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for modern premium spacing and typography
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f111a;
    }
    .metric-card {
        background-color: #1e2230;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e3440;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #88c0d0;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #d8dee9;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Classical Time Series Diagnostics Dashboard")
st.markdown("""
Welcome to the diagnostics studio for your Ecuador Store Sales forecasting project. This tool helps you dissect the performance of **SARIMAX**, **Prophet**, and **ARX-GARCH**, pinpointing exactly *where* they fail and diagnosing whether they suffer from **under/over-forecasting bias** or **volatility smoothing**. 

Use the insights below to guide your feature engineering and model selection as you transition to machine learning.
""")

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('classical_timeseries_evaluation_metrics.csv')
    
    # Calculate Prediction Bias (Positive = Overforecasting, Negative = Underforecasting)
    df['bias'] = df['pred_mean'] - df['true_mean_sales']
    df['abs_bias'] = df['bias'].abs()
    
    # Calculate Volatility Ratio (Predicted Variance / True Variance)
    # Avoid division by zero when true variance is exactly 0 (e.g. all zero sales)
    df['volatility_ratio'] = np.where(
        df['true_variance_sales'] > 0, 
        df['pred_variance'] / df['true_variance_sales'], 
        np.nan
    )
    
    # Category sales volume based on true mean sales
    df['volume_tier'] = pd.cut(
        df['true_mean_sales'], 
        bins=[-1, 10, 100, float('inf')], 
        labels=['Low (<10)', 'Medium (10-100)', 'High (100+)']
    )
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading the evaluation metrics file: {e}")
    st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🎯 Filter Workspace")
st.sidebar.markdown("Slice and dice the time series data to investigate specific subsets.")

# Store filter
stores = sorted(df['store_nbr'].unique())
selected_stores = st.sidebar.multiselect(
    'Select Store(s)', 
    stores, 
    default=stores[:3],
    help="Filter by specific supermarket store numbers in Ecuador."
)

# Product Family filter
families = sorted(df['family'].unique())
selected_families = st.sidebar.multiselect(
    'Select Product Family', 
    families, 
    default=families[:5],
    help="Filter by product categories (e.g., GROCERY I, BEVERAGES, AUTOMOTIVE)."
)

# Stationarity filter
stationarity_opt = st.sidebar.radio(
    "Stationarity (ADF Test)", 
    ["All", "Stationary Only", "Non-Stationary Only"],
    help="Filter by whether the time series was detected as stationary by the Augmented Dickey-Fuller test."
)

# Sales Volume filter
volume_opt = st.sidebar.selectbox(
    "Sales Volume Tier", 
    ["All", "Low (<10 average sales)", "Medium (10-100 average sales)", "High (100+ average sales)"],
    help="Filter series by their average daily sales volume. Helps isolate high-volume drivers from intermittent demand."
)

# Active Metric for Visuals
active_metric = st.sidebar.radio(
    "Primary Error Metric",
    ["RMSLE", "MAE"],
    help="RMSLE penalizes underpredictions more than overpredictions on log-scale; MAE provides a direct linear error scale."
)

# Apply filters to DataFrame
filtered_df = df[
    (df['store_nbr'].isin(selected_stores)) & 
    (df['family'].isin(selected_families))
]

if stationarity_opt == "Stationary Only":
    filtered_df = filtered_df[filtered_df['is_stationary'] == True]
elif stationarity_opt == "Non-Stationary Only":
    filtered_df = filtered_df[filtered_df['is_stationary'] == False]

if volume_opt == "Low (<10 average sales)":
    filtered_df = filtered_df[filtered_df['volume_tier'] == 'Low (<10)']
elif volume_opt == "Medium (10-100 average sales)":
    filtered_df = filtered_df[filtered_df['volume_tier'] == 'Medium (10-100)']
elif volume_opt == "High (100+ average sales)":
    filtered_df = filtered_df[filtered_df['volume_tier'] == 'High (100+)']

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
if filtered_df.empty:
    st.warning("⚠️ No data matches your active filters. Please adjust the sidebar settings.")
else:
    # Set custom color palette for consistent model representation
    model_colors = {
        'SARIMAX': '#3b82f6',   # Blue
        'Prophet': '#f97316',   # Orange
        'ARX-GARCH': '#a855f7'  # Purple
    }

    # Define Tabs
    tab_perf, tab_bias, tab_vol, tab_segment, tab_fail, tab_ml = st.tabs([
        "📊 Performance Overview", 
        "⚖️ Overcasting & Bias", 
        "🌪️ Volatility & Variance", 
        "🧩 Segmented Diagnostics", 
        "🕵️ Failure Explorer", 
        "🔮 ML Transition Blueprint"
    ])

    # --------------------------------------------------
    # TAB 1: PERFORMANCE OVERVIEW
    # --------------------------------------------------
    with tab_perf:
        st.subheader("🏆 Model Performance Comparison")
        
        # --- Top Row KPIs ---
        col1, col2, col3, col4 = st.columns(4)
        
        num_series = filtered_df.groupby(['store_nbr', 'family']).ngroups
        best_model = filtered_df.groupby('model_name')[active_metric.lower()].mean().idxmin()
        avg_rmsle = filtered_df[filtered_df['model_name'] == best_model]['rmsle'].mean()
        percent_stationary = (filtered_df['is_stationary'].mean()) * 100
        
        col1.metric("Selected Series (Store-Family Keys)", f"{num_series}")
        col2.metric(f"Best Model (Lowest Mean {active_metric})", best_model)
        col3.metric(f"Average RMSLE ({best_model})", f"{avg_rmsle:.4f}")
        col4.metric("Stationary Series Rate", f"{percent_stationary:.1f}%")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🥇 Model Win Rates")
            st.markdown("What percentage of individual time series does each model perform best on?")
            
            # Calculate win rates for each series
            idx = filtered_df.groupby(['store_nbr', 'family'])[active_metric.lower()].idxmin()
            best_models_per_series = filtered_df.loc[idx]
            win_rates = best_models_per_series['model_name'].value_counts(normalize=True) * 100
            
            # Fill 0% for missing models in active slice
            for m in model_colors.keys():
                if m not in win_rates:
                    win_rates[m] = 0.0
            win_rates = win_rates.reindex(model_colors.keys()).reset_index()
            win_rates.columns = ['Model', 'Win Percentage']
            
            fig_win = px.bar(
                win_rates,
                x='Win Percentage',
                y='Model',
                orientation='h',
                color='Model',
                color_discrete_map=model_colors,
                title=f"Win Rate per Series (Based on {active_metric})",
                text=win_rates['Win Percentage'].apply(lambda x: f"{x:.1f}%")
            )
            fig_win.update_layout(showlegend=False, xaxis_title="Percentage of Series Won (%)", yaxis_title="")
            st.plotly_chart(fig_win, use_container_width=True)
            
        with col_right:
            st.markdown(f"### 📦 {active_metric} Distribution")
            st.markdown("Compare the spread and outlier counts of forecasting errors across models.")
            
            fig_dist = px.box(
                filtered_df,
                x='model_name',
                y=active_metric.lower(),
                color='model_name',
                color_discrete_map=model_colors,
                title=f"{active_metric} Distribution by Model",
                labels={'model_name': 'Model', 'rmsle': 'RMSLE', 'mae': 'MAE'}
            )
            fig_dist.update_layout(showlegend=False, xaxis_title="", yaxis_title=active_metric)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        st.markdown("### 📋 Aggregated Summary Stats")
        summary_stats = filtered_df.groupby('model_name').agg(
            Mean_RMSLE=('rmsle', 'mean'),
            Median_RMSLE=('rmsle', 'median'),
            Mean_MAE=('mae', 'mean'),
            Median_MAE=('mae', 'median'),
            Std_RMSLE=('rmsle', 'std')
        ).reindex(model_colors.keys()).reset_index()
        
        st.dataframe(
            summary_stats.style.format({
                'Mean_RMSLE': '{:.4f}',
                'Median_RMSLE': '{:.4f}',
                'Mean_MAE': '{:.2f}',
                'Median_MAE': '{:.2f}',
                'Std_RMSLE': '{:.4f}'
            }),
            use_container_width=True
        )

    # --------------------------------------------------
    # TAB 2: OVERCASTING & BIAS ("Am I Overcasting?")
    # --------------------------------------------------
    with tab_bias:
        st.subheader("⚖️ Overforecasting vs. Underforecasting Diagnostics")
        
        st.info("""
        💡 **Understanding Prediction Bias**: 
        * **Bias = Predicted Mean - True Mean**.
        * **Positive Bias (> 0)** indicates **Overforecasting** (overcasting). Overforecasting inflates inventory levels, raising carrying costs and risking waste.
        * **Negative Bias (< 0)** indicates **Underforecasting**. Underforecasting leads to stockouts, empty shelves, and lost revenue opportunities.
        """)
        
        # Calculate overforecast rate
        filtered_df['is_overforecast'] = filtered_df['bias'] > 0
        
        bias_kpis = filtered_df.groupby('model_name').agg(
            over_pct=('is_overforecast', 'mean'),
            mean_bias=('bias', 'mean'),
            median_bias=('bias', 'median')
        ).reindex(model_colors.keys())
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Column 1: SARIMAX
        with col_m1:
            st.markdown("#### 🔹 SARIMAX")
            st.metric("Overforecasting Rate", f"{bias_kpis.loc['SARIMAX', 'over_pct']*100:.1f}%")
            st.metric("Mean Bias (Sales)", f"{bias_kpis.loc['SARIMAX', 'mean_bias']:.2f}")
            st.metric("Median Bias (Sales)", f"{bias_kpis.loc['SARIMAX', 'median_bias']:.2f}")
            
        # Column 2: Prophet
        with col_m2:
            st.markdown("#### 🔸 Prophet")
            st.metric("Overforecasting Rate", f"{bias_kpis.loc['Prophet', 'over_pct']*100:.1f}%")
            st.metric("Mean Bias (Sales)", f"{bias_kpis.loc['Prophet', 'mean_bias']:.2f}")
            st.metric("Median Bias (Sales)", f"{bias_kpis.loc['Prophet', 'median_bias']:.2f}")
            
        # Column 3: ARX-GARCH
        with col_m3:
            st.markdown("#### 🔮 ARX-GARCH")
            st.metric("Overforecasting Rate", f"{bias_kpis.loc['ARX-GARCH', 'over_pct']*100:.1f}%")
            st.metric("Mean Bias (Sales)", f"{bias_kpis.loc['ARX-GARCH', 'mean_bias']:.2f}")
            st.metric("Median Bias (Sales)", f"{bias_kpis.loc['ARX-GARCH', 'median_bias']:.2f}")
            
        st.markdown("---")
        
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            st.markdown("### 📊 Distribution of Prediction Bias")
            st.markdown("A wider distribution means unpredictable, volatile forecast errors. Check if the model distribution centers on zero or skews positive/negative.")
            fig_bias_dist = px.box(
                filtered_df,
                x='model_name',
                y='bias',
                color='model_name',
                color_discrete_map=model_colors,
                title="Bias Spread by Model (Predicted - True Mean Sales)",
                labels={'model_name': 'Model', 'bias': 'Bias Value (Daily Sales)'}
            )
            # Add horizontal line at 0 bias
            fig_bias_dist.add_shape(
                type="line", x0=-0.5, y0=0, x1=2.5, y1=0,
                line=dict(color="red", width=2, dash="dash")
            )
            fig_bias_dist.update_layout(showlegend=False)
            st.plotly_chart(fig_bias_dist, use_container_width=True)
            
        with col_b2:
            st.markdown("### 🔍 Bias vs. Sales Volume (The Scale Problem)")
            st.markdown("Do models overforecast low-volume series and underforecast high-volume series?")
            fig_bias_scale = px.scatter(
                filtered_df,
                x='true_mean_sales',
                y='bias',
                color='model_name',
                color_discrete_map=model_colors,
                hover_data=['store_nbr', 'family'],
                title="Bias vs. True Average Sales",
                labels={'true_mean_sales': 'True Average Sales', 'bias': 'Bias'}
            )
            fig_bias_scale.add_shape(
                type="line", x0=0, y0=0, x1=filtered_df['true_mean_sales'].max(), y1=0,
                line=dict(color="red", width=2, dash="dash")
            )
            st.plotly_chart(fig_bias_scale, use_container_width=True)
            
        st.markdown("""
        💡 **Data Science Takeaway on Bias**:
        * **Low-Volume items (intermittent sales)** often show **positive bias** (overforecasting). This occurs because classical models predict a continuous positive sales flow (e.g. 0.4 units daily) rather than modeling discrete transactions, inflating predicted averages above true sales.
        * **High-Volume items** frequently show **negative bias** (underforecasting). Classical linear models fail to capture large promotional demand spikes (which are typically dynamic and calendar-dependent) unless provided with complex exogenous variables, smoothing out these peaks and leaving a deficit.
        """)

    # --------------------------------------------------
    # TAB 3: VOLATILITY & VARIANCE CAPTURE
    # --------------------------------------------------
    with tab_vol:
        st.subheader("🌪️ Volatility & Variance Capture Analysis")
        
        st.info("""
        💡 **Understanding the Volatility Ratio (`pred_variance / true_variance`)**:
        * **Ratio = 1.0**: Perfect variance representation. The model matches the variability of actual retail sales.
        * **Ratio < 1.0**: **Variance Underestimation (Smoothing)**. The model predicts a smooth average trend and completely misses peak promotional spikes, payday cycles, and holiday events.
        * **Ratio > 1.0**: **Overreacting**. The model predicts unstable, explosive, or noisy forecasts.
        """)
        
        vol_kpis = filtered_df.groupby('model_name')['volatility_ratio'].median().reindex(model_colors.keys())
        
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("SARIMAX Median Volatility Ratio", f"{vol_kpis.loc['SARIMAX']:.4f}")
        col_v2.metric("Prophet Median Volatility Ratio", f"{vol_kpis.loc['Prophet']:.4f}")
        col_v3.metric("ARX-GARCH Median Volatility Ratio", f"{vol_kpis.loc['ARX-GARCH']:.4f}")
        
        st.markdown("---")
        
        col_vplot1, col_vplot2 = st.columns(2)
        
        with col_vplot1:
            st.markdown("### 📈 Volatility Ratio Distribution")
            st.markdown("Distribution of volatility capture ratios. Values below 1.0 indicate forecasting smoothing.")
            
            # Filter out NaNs and infs for visualization
            valid_vol_df = filtered_df[filtered_df['volatility_ratio'].notna() & (filtered_df['volatility_ratio'] > 0)]
            
            fig_vol_box = px.box(
                valid_vol_df,
                x='model_name',
                y='volatility_ratio',
                color='model_name',
                color_discrete_map=model_colors,
                log_y=True,
                title="Volatility Ratio Distribution (Log Scale)",
                labels={'model_name': 'Model', 'volatility_ratio': 'Volatility Ratio'}
            )
            # Add line at 1.0 (perfect variance capture)
            fig_vol_box.add_shape(
                type="line", x0=-0.5, y0=1.0, x1=2.5, y1=1.0,
                line=dict(color="red", width=2, dash="dash")
            )
            fig_vol_box.update_layout(showlegend=False)
            st.plotly_chart(fig_vol_box, use_container_width=True)
            
        with col_vplot2:
            st.markdown("### 🔍 True Variance vs. Predicted Variance")
            st.markdown("Do models lose tracking accuracy on highly volatile series?")
            
            fig_var_scatter = px.scatter(
                filtered_df,
                x='true_variance_sales',
                y='pred_variance',
                color='model_name',
                color_discrete_map=model_colors,
                hover_data=['store_nbr', 'family'],
                log_x=True,
                log_y=True,
                title="True Variance vs. Predicted Variance (Log-Log)",
                labels={'true_variance_sales': 'True Variance', 'pred_variance': 'Predicted Variance'}
            )
            max_var = max(filtered_df['true_variance_sales'].max(), filtered_df['pred_variance'].max())
            fig_var_scatter.add_shape(
                type="line", x0=1, y0=1, x1=max_var, y1=max_var,
                line=dict(color="red", width=2, dash="dash")
            )
            st.plotly_chart(fig_var_scatter, use_container_width=True)

        st.markdown("""
        💡 **Data Science Takeaway on Volatility**:
        * **Prophet** and **SARIMAX** often smooth time series dramatically when they cannot incorporate dynamic regressors (like daily store promotions). Their volatility ratios are typically far below 1.0.
        * **ARX-GARCH** models volatility directly (conditional heteroskedasticity). However, if GARCH is fit on daily sales containing huge zero-inflated regions or high-frequency promotional noise, it can either break down or underforecast severely (as shown in the Bias tab) to satisfy stationary likelihood constraints.
        """)

# --------------------------------------------------
    # TAB 4: SEGMENTED DIAGNOSTICS ("Where are models failing?")
    # --------------------------------------------------
    with tab_segment:
        st.subheader("🧩 Segmented Performance Analysis")
        st.markdown("Drill down to see which features of a time series dictate performance.")
        
        # --- Section 1: Stationarity & Volume ---
        col_seg1, col_seg2 = st.columns(2)
        
        with col_seg1:
            st.markdown("#### 🗺️ Performance by Stationarity (ADF Test)")
            stationarity_breakdown = filtered_df.groupby(['is_stationary', 'model_name'], observed=False).agg(
                Avg_RMSLE=('rmsle', 'mean'),
                Avg_Bias=('bias', 'mean'),
                Count=('rmsle', 'count')
            ).reset_index()
            
            st.dataframe(
                stationarity_breakdown.style.format({
                    'Avg_RMSLE': '{:.4f}',
                    'Avg_Bias': '{:.2f}'
                }),
                use_container_width=True
            )
            
        with col_seg2:
            st.markdown("#### 📈 Performance by Sales Volume Tier")
            volume_breakdown = filtered_df.groupby(['volume_tier', 'model_name'], observed=False).agg(
                Avg_RMSLE=('rmsle', 'mean'),
                Avg_Bias=('bias', 'mean'),
                Count=('rmsle', 'count')
            ).reset_index()
            
            st.dataframe(
                volume_breakdown.style.format({
                    'Avg_RMSLE': '{:.4f}',
                    'Avg_Bias': '{:.2f}'
                }),
                use_container_width=True
            )
            
        st.markdown("---")
        
        # --- Section 2: Product Family Bar Chart ---
        st.markdown("### 🛒 Performance Breakdown by Product Family")
        st.markdown("Identify which product families are easy to model and which ones are highly problematic.")
        
        family_perf = filtered_df.groupby(['family', 'model_name']).agg(
            metric_val=(active_metric.lower(), 'mean')
        ).reset_index()
        
        fig_fam = px.bar(
            family_perf,
            x='family',
            y='metric_val',
            color='model_name',
            color_discrete_map=model_colors,
            barmode='group', # Grouped bars
            title=f"Average {active_metric} by Product Family",
            labels={'metric_val': f'Average {active_metric}', 'family': 'Product Family'}
        )
        fig_fam.update_layout(barmode='group', xaxis_tickangle=-45)
        st.plotly_chart(fig_fam, use_container_width=True)
        
        # --- Section 3: Store Performance Bar Chart ---
        st.markdown("### 🏪 Performance Breakdown by Store")
        st.markdown("Are some stores inherently harder to forecast due to local volatility or size?")
        
        store_perf = filtered_df.groupby(['store_nbr', 'model_name']).agg(
            metric_val=(active_metric.lower(), 'mean')
        ).reset_index()
        store_perf['store_nbr'] = store_perf['store_nbr'].astype(str) # Convert to categorical for discrete x-axis
        
        fig_store = px.bar(
            store_perf,
            x='store_nbr',
            y='metric_val',
            color='model_name',
            color_discrete_map=model_colors,
            title=f"Average {active_metric} by Store Number",
            labels={'metric_val': f'Average {active_metric}', 'store_nbr': 'Store Number'}
        )
        fig_store.update_layout(barmode='group')
        st.plotly_chart(fig_store, use_container_width=True)

    # --------------------------------------------------
    # TAB 5: MODEL FAILURE EXPLORER
    # --------------------------------------------------
    with tab_fail:
        st.subheader("🕵️ Deep Dive: worst performing time series")
        st.markdown("Identify and examine the exact store-product family pairs where models suffered their largest failures.")
        
        selected_fail_model = st.selectbox(
            "Select Model to Investigate", 
            ["SARIMAX", "Prophet", "ARX-GARCH"]
        )
        
        model_failures = filtered_df[filtered_df['model_name'] == selected_fail_model]
        worst_failures = model_failures.sort_values(by=active_metric.lower(), ascending=False).head(15)
        
        st.dataframe(
            worst_failures[[
                'store_nbr', 'family', 'rmsle', 'mae', 'is_stationary', 
                'true_mean_sales', 'pred_mean', 'bias', 
                'true_variance_sales', 'pred_variance', 'volatility_ratio'
            ]].style.format({
                'rmsle': '{:.4f}',
                'mae': '{:.2f}',
                'true_mean_sales': '{:.2f}',
                'pred_mean': '{:.2f}',
                'bias': '{:.2f}',
                'true_variance_sales': '{:.2f}',
                'pred_variance': '{:.2f}',
                'volatility_ratio': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Display failure insights depending on model
        st.markdown("### 💡 Failure Mode Assessment")
        if selected_fail_model == "SARIMAX":
            st.markdown("""
            * **Local Non-stationarity**: SARIMAX depends heavily on correct differencing ($d$). If the series has sudden structural breaks (e.g. store openings, earthquake shocks in Ecuador 2016), local SARIMAX equations fail and experience massive error drifting.
            * **Linear Exogenous Bottleneck**: SARIMAX assumes linear relationships. It can't easily capture compound relationships like 'promotion ON plus payday weekend'.
            * **No Negative Sales Clipping**: SARIMAX can forecast negative sales on low-volume, zero-inflated items. Ifclipped to 0, bias is introduced, leading to inflated RMSLE.
            """)
        elif selected_fail_model == "Prophet":
            st.markdown("""
            * **Over-smoothing Local Shocks**: Prophet models additive/multiplicative trends with Fourier terms for seasonality. While excellent for broad national retail seasonalities, it severely smooths out short-term, local shocks like promotion days unless specifically configured with an intensive, custom holiday dataframe.
            * **Low-Volume Intermittent Bias**: Prophet operates on continuous distributions. For items like 'BABY CARE' or 'BOOKS' that sell 0 units on most days, Prophet often fits a smooth decimal curve (e.g., 0.3 units daily), leading to high overforecasting rates.
            """)
        elif selected_fail_model == "ARX-GARCH":
            st.markdown("""
            * **Severely Out-of-Scale forecasts**: ARX-GARCH shows high negative bias in our analysis. GARCH equations are extremely sensitive to conditional variance stability. If sales jump from 0 to 1000 due to a single mega-promotion, GARCH models struggle with the extreme 'volatility cluster' and either explode or shrink their mean estimates to enforce model stationarity.
            * **Inefficient on Low-Volume Series**: Modeling autoregressive conditional heteroskedasticity is mathematically unstable on intermittent, zero-heavy sales data.
            """)

    # --------------------------------------------------
    # TAB 6: ML TRANSITION BLUEPRINT
    # --------------------------------------------------
    with tab_ml:
        st.subheader("🔮 Machine Learning Transition Blueprint")
        st.markdown("How to leverage the failures of classical models to design a winning Kaggle Machine Learning pipeline.")
        
        # Calculate dynamic insights on filtered dataset
        pct_non_stationary = (1 - filtered_df['is_stationary'].mean()) * 100
        pct_low_vol = (filtered_df['true_mean_sales'] < 10).mean() * 100
        
        median_vol_sarimax = filtered_df[filtered_df['model_name'] == 'SARIMAX']['volatility_ratio'].median()
        median_vol_prophet = filtered_df[filtered_df['model_name'] == 'Prophet']['volatility_ratio'].median()
        
        # Recommendation 1: Global vs Local
        st.markdown(f"""
        ### ⚖️ Core Architectural Shift: Global vs. Local Models
        
        Currently, you are fitting **local models**—meaning you must fit a distinct SARIMAX, Prophet, or ARX-GARCH model for **each individual store and product family combinations** (currently `{num_series}` separate models in this filtered slice!).
        
        **Why this is a massive bottleneck:**
        1. **Computational Overhead**: Fitting thousands of models is slow, expensive, and scales horribly.
        2. **Isolation of Information**: The model for *Store 1, AUTOMOTIVE* has no idea what is happening in *Store 2, AUTOMOTIVE*, or what the general product category trend is.
        
        👉 **ML Recommendation**: Transition to a **Global Model** (e.g. a single LightGBM, XGBoost, CatBoost, or DeepAR model). Global models train on a single combined dataset. They use entity embeddings or categorical columns (e.g. `store_nbr`, `family`, `store_cluster`, `store_type`) to learn a master representation of supermarket sales, transferring patterns (like weekend effects or holiday demand) across all stores and product families!
        """)
        
        st.markdown("---")
        
        # Dynamic Diagnostic 2: Volatility
        st.markdown("### 🌪️ Solving Volatility Smoothing")
        if (median_vol_sarimax < 0.6) or (median_vol_prophet < 0.6):
            st.warning(f"""
            ⚠️ **Diagnostic**: Your classical models are severely smoothing sales spikes (SARIMAX captures only {median_vol_sarimax*100:.1f}% and Prophet captures only {median_vol_prophet*100:.1f}% of actual variance). They cannot track Ecuador's highly volatile promotion days, oil price shocks, and national holidays.
            """)
        else:
            st.info("""
            ℹ️ **Diagnostic**: Your classical models are capturing variance reasonably well in this slice, but still lack dynamic flexibility.
            """)
            
        st.markdown("""
        👉 **ML Strategy: Feature-Rich Tree Ensembles & Attention Neural Nets**
        1. **Tree Ensembles (LightGBM/XGBoost/CatBoost)**: Tree models excel at step-changes and sharp peaks. Feed them with:
            * **Lag Features**: `sales_lag_16` (to forecast a 16-day horizon), `sales_lag_17`, etc.
            * **Rolling Statistics**: Mean, standard deviation, and sum of sales over 7, 14, and 30-day windows.
            * **Dynamic Regressors**: Promotion flag (`onpromotion`), daily oil price, store transaction volumes, and categorical holiday encoders.
        2. **Attention Architectures (Temporal Fusion Transformer - TFT)**:
            * TFT natively separates 'known future inputs' (promotions, holiday calendars, payday schedules) from 'observed historical inputs' (lags of sales, oil prices) to learn complex attention-weighted peak timings.
        """)
        
        st.markdown("---")
        
        # Dynamic Diagnostic 3: Zero-Inflation
        st.markdown("### 🎯 Solving Intermittent Demand (Low Volume)")
        if pct_low_vol > 30:
            st.warning(f"⚠️ **Diagnostic**: {pct_low_vol:.1f}% of your filtered series are **Low-Volume (<10 average sales)**. Normal Gaussian loss functions (MSE/RMSE) will bias models to overpredict zero-heavy zones.")
        else:
            st.info(f"ℹ️ **Diagnostic**: Low-volume intermittent demand constitutes {pct_low_vol:.1f}% of this filtered slice.")
            
        st.markdown("""
        👉 **ML Strategy: Custom Loss Objectives & Count-Data Modeling**
        1. **Tweedie Loss or Poisson Loss**: 
            * When training LightGBM, XGBoost, or CatBoost, change your objective parameter to `objective='tweedie'` or `objective='poisson'`. This optimizes the model for zero-inflated, highly skewed continuous/count variables.
        2. **Zero-Inflated Neural Emission (DeepAR)**:
            * DeepAR (GluonTS) models the target variable probabilistically. Change its emission distribution parameter to a **Negative Binomial** or **Zero-Inflated Poisson (ZIP)** distribution. This allows the model to predict exact probability densities for 0 sales, preventing continuous positive bias.
        """)
        
        st.markdown("---")
        
        # Dynamic Diagnostic 4: Stationarity
        st.markdown("### 🗺️ Handling Trend & Non-Stationarity")
        if pct_non_stationary > 30:
            st.warning(f"⚠️ **Diagnostic**: {pct_non_stationary:.1f}% of your filtered series are **Non-Stationary**. Tree-based models have a critical weakness: they **cannot extrapolate trends**.")
        else:
            st.info(f"ℹ️ **Diagnostic**: Non-stationary series constitute {pct_non_stationary:.1f}% of this filtered slice.")
            
        st.markdown(r"""
        👉 **ML Strategy: Detrending, Hybrids, and Temporal Architectures**
        1. **Differencing & Target Transformation**:
            * Fit your tree models on log-differenced sales: $y'_t = \log(y_t + 1) - \log(y_{t-7} + 1)$ (to remove weekly trend), then reverse the transformation in post-processing.
        2. **Hybrid Modeling (Trend + Residual)**:
            * Use **Prophet or Linear Regression** to model and forecast the long-term trend.
            * Use **LightGBM** to train on and forecast the *residuals* (errors of the trend model). This combines the trend-extrapolation strength of linear models with the high-variance capture strength of decision trees!
        3. **Dedicated Deep Learning**:
            * **DLinear / PatchTST / N-BEATS**: These modern neural forecasting architectures have explicit, structural linear and trend-seasonal decomposition blocks designed specifically to handle trend extrapolation natively.
        """)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(page_title="Supermarket Forecasting EDA", layout="wide")

st.title("🛒 Classical Time Series Evaluation Dashboard")
st.markdown("Analyze how SARIMAX, Prophet, and ARX-GARCH handle different stores and product families in Ecuador.")

# Load Data efficiently
@st.cache_data
def load_data():
    df = pd.read_csv('classical_timeseries_evaluation_metrics.csv')
    df['bias'] = df['pred_mean'] - df['true_mean_sales']
    return df

df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Filter Data")

# Filter by Store
stores = sorted(df['store_nbr'].unique())
selected_stores = st.sidebar.multiselect('Select Store(s)', stores, default=stores[:3])

# Filter by Family
families = sorted(df['family'].unique())
selected_families = st.sidebar.multiselect('Select Product Family', families, default=families[:5])

# Filter by Stationarity
stationarity = st.sidebar.radio("Stationarity", ["All", "Stationary Only", "Non-Stationary Only"])

# Apply Filters
filtered_df = df[
    (df['store_nbr'].isin(selected_stores)) & 
    (df['family'].isin(selected_families))
]

if stationarity == "Stationary Only":
    filtered_df = filtered_df[filtered_df['is_stationary'] == True]
elif stationarity == "Non-Stationary Only":
    filtered_df = filtered_df[filtered_df['is_stationary'] == False]

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
if filtered_df.empty:
    st.warning("No data available for these filters.")
else:
    # --- Top Row KPIs ---
    col1, col2, col3 = st.columns(3)
    
    best_model = filtered_df.groupby('model_name')['rmsle'].mean().idxmin()
    avg_rmsle = filtered_df['rmsle'].mean()
    percent_stationary = (filtered_df['is_stationary'].mean()) * 100
    
    col1.metric("Overall Best Model (Lowest RMSLE)", best_model)
    col2.metric("Average RMSLE (Filtered)", f"{avg_rmsle:.4f}")
    col3.metric("Stationary Timeseries", f"{percent_stationary:.1f}%")
    
    st.markdown("---")
    
    # --- Row 2: Model Performance Comparison ---
    st.subheader("Model Performance Comparison")
    
    # Bar Chart: RMSLE by Model
    fig_rmsle = px.box(
        filtered_df, 
        x='model_name', 
        y='rmsle', 
        color='model_name',
        title="RMSLE Distribution by Model",
        labels={'model_name': 'Model', 'rmsle': 'RMSLE'}
    )
    st.plotly_chart(fig_rmsle, use_container_width=True)

    # --- Row 3: Deep Dive Scatters ---
    st.subheader("Forecast Behavior: Bias and Volatility")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Scatter: Mean Bias (Over vs Under Forecasting)
        fig_bias = px.scatter(
            filtered_df, 
            x='true_mean_sales', 
            y='pred_mean', 
            color='model_name',
            hover_data=['store_nbr', 'family'],
            title="True Mean vs. Predicted Mean",
            labels={'true_mean_sales': 'True Average Sales', 'pred_mean': 'Predicted Average Sales'}
        )
        # Add diagonal perfect prediction line
        max_mean = max(filtered_df['true_mean_sales'].max(), filtered_df['pred_mean'].max())
        fig_bias.add_shape(type="line", x0=0, y0=0, x1=max_mean, y1=max_mean, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_bias, use_container_width=True)

    with col_b:
        # Scatter: Variance Capture
        fig_var = px.scatter(
            filtered_df, 
            x='true_variance_sales', 
            y='pred_variance', 
            color='model_name',
            hover_data=['store_nbr', 'family'],
            log_x=True, log_y=True,
            title="True Variance vs. Predicted Variance (Log Scale)",
            labels={'true_variance_sales': 'True Variance', 'pred_variance': 'Predicted Variance'}
        )
        max_var = max(filtered_df['true_variance_sales'].max(), filtered_df['pred_variance'].max())
        fig_var.add_shape(type="line", x0=1, y0=1, x1=max_var, y1=max_var, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_var, use_container_width=True)

    # --- Row 4: Raw Data Table ---
    st.markdown("---")
    st.subheader("Detailed Metrics Table")
    st.dataframe(
        filtered_df[['store_nbr', 'family', 'model_name', 'rmsle', 'mae', 'is_stationary', 'bias']]
        .sort_values(by=['store_nbr', 'family', 'rmsle']),
        use_container_width=True
    )

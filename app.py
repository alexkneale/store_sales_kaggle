import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

st.set_page_config(page_title="Store Sales EDA", layout="wide", page_icon="🛒")

# ==========================================
# 1. CUSTOM PIPELINE DEFINITION
# ==========================================
class StoreSalesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, oil_df, stores_df, holidays_df=None, transactions_df=None):
        self.oil_df = oil_df.copy()
        self.stores_df = stores_df.copy()
        self.holidays_df = holidays_df.copy() if holidays_df is not None else None
        self.transactions_df = transactions_df.copy() if transactions_df is not None else None
        
    def fit(self, X, y=None):
        self.processed_oil_ = self._process_oil(self.oil_df)
        self.processed_stores_ = self._process_stores(self.stores_df)
        self.processed_holidays_ = self._process_holidays(self.holidays_df)
        self.process_transactions_ = self._process_transactions(transactions=self.transactions_df)
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['date'] = pd.to_datetime(X_out['date'])
        
        X_out = pd.merge(X_out, self.processed_oil_, on='date', how='left')
        X_out = pd.merge(X_out, self.processed_stores_, on='store_nbr', how='left')
        X_out = pd.merge(X_out, self.processed_holidays_, on='date', how='left')
        X_out = pd.merge(X_out, self.process_transactions_, on=['date', 'store_nbr'], how='left')

        
        X_out['day_of_week'] = X_out['date'].dt.dayofweek
        # day of month
        X_out['day_of_month'] = X_out['date'].dt.day
        # is last day of the month
        X_out['is_pay_day'] = X_out['date'].dt.day.isin([15]).astype(int) | X_out['date'].dt.is_month_end

        X_out['month'] = X_out['date'].dt.month
        X_out['year'] = X_out['date'].dt.year
        X_out['is_weekend'] = X_out['day_of_week'].isin([5, 6]).astype(int)
        
        return X_out

    def _process_oil(self, oil):
        oil['date'] = pd.to_datetime(oil['date'])
        calendar = pd.date_range(start=oil['date'].min(), end=oil['date'].max())
        oil_continuous = oil.set_index('date').reindex(calendar).rename_axis('date').reset_index()
        oil_continuous['dcoilwtico'] = oil_continuous['dcoilwtico'].ffill().bfill()
        oil_continuous = oil_continuous.rename(columns={'dcoilwtico': 'oil_price'})
        return oil_continuous
    
    def _process_holidays(self, holidays):
        holidays['date'] = pd.to_datetime(holidays['date'])
        holidays = holidays.rename(columns={
            'locale': 'holiday_location', 
            'locale_name': 'holiday_location_name', 
            'type': 'holiday_type', 
            'description': 'holiday_description',  
            'transferred': 'holiday_transferred'
        })
        return holidays
    
    def _process_stores(self, stores):
        stores = stores.rename(columns={
            'city': 'store_city', 
            'state': 'store_state', 
            'type': 'store_type', 
            'cluster': 'store_cluster'
        })
        return stores

    def _process_transactions(self, transactions):
        transactions['date'] = pd.to_datetime(transactions['date'])
        return transactions

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    # Load raw data based on your directory structure
    path = 'store-sales-time-series-forecasting/'
    train_raw = pd.read_csv(f'{path}train.csv')
    oil_raw = pd.read_csv(f'{path}oil.csv')
    stores_raw = pd.read_csv(f'{path}stores.csv')
    holidays_raw = pd.read_csv(f'{path}holidays_events.csv')
    transactions_raw = pd.read_csv(f'{path}transactions.csv')


    # Initialize and run pipeline
    feature_pipeline = Pipeline([
        ('preprocessor', StoreSalesPreprocessor(oil_df=oil_raw, stores_df=stores_raw, holidays_df=holidays_raw, transactions_df=transactions_raw))
    ])
    
    df = feature_pipeline.fit_transform(train_raw)
    
    # Add weekly and daily calendar components for the visualizations
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Fill missing holiday values with 'None' so grouping works nicely
    df['holiday_type'] = df['holiday_type'].fillna('No Holiday')
    
    return df

with st.spinner("Running Sklearn Pipeline and processing 3M rows..."):
    df = load_data()

# ==========================================
# 3. DASHBOARD UI & FILTERS
# ==========================================
st.sidebar.title("Filters")
selected_stores = st.sidebar.multiselect("Select Stores", options=df['store_nbr'].unique(), default=[])
selected_families = st.sidebar.multiselect("Select Product Families", options=df['family'].unique(), default=[])

filtered_df = df.copy()
if selected_stores:
    filtered_df = filtered_df[filtered_df['store_nbr'].isin(selected_stores)]
if selected_families:
    filtered_df = filtered_df[filtered_df['family'].isin(selected_families)]

st.title("🛒 Store Sales - Pipeline Augmented EDA")
st.markdown("Explore trends, seasonality, external factors (Oil/Holidays), and autocorrelations.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Time Series (YoY)", "🏬 Stores & Families", "🛢️ Oil & Holidays", "🔄 ACF/PACF", "📢 Promotions", "💳 Transactions"])

# ==========================================
# TAB 1: TIME SERIES (Year-on-Year)
# ==========================================
with tab1:
    st.header("Time Series Seasonality (Year on Year)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yearly Average Sales")
        yearly_sales = filtered_df.groupby('year')['sales'].mean().reset_index()
        fig_year = px.bar(yearly_sales, x='year', y='sales', text_auto='.2s', color='year')
        st.plotly_chart(fig_year, width='stretch')

        st.subheader("Weekly Average Sales (YoY)")
        weekly_sales = filtered_df.groupby(['year', 'week'])['sales'].mean().reset_index()
        fig_week = px.line(weekly_sales, x='week', y='sales', color='year', markers=True)
        st.plotly_chart(fig_week, width='stretch')

        st.subheader("Average Sales by Day of the Week (YoY)")
        weekly_sales = filtered_df.groupby(['year', 'day_of_week'])['sales'].mean().reset_index()
        fig_week = px.line(weekly_sales, x='day_of_week', y='sales', color='year', markers=True)
        st.plotly_chart(fig_week, width='stretch')

        # average sales payday vs non payday
    
        st.subheader("Average Sales by Payday vs non payday")
        payday_sales = filtered_df.groupby(['year', 'is_pay_day'])['sales'].mean().reset_index()
        fig_week = px.line(payday_sales, x='is_pay_day', y='sales', color='year', markers=True)
        st.plotly_chart(fig_week, width='stretch')

    with col2:
        st.subheader("Monthly Average Sales (YoY)")
        monthly_sales = filtered_df.groupby(['year', 'month'])['sales'].mean().reset_index()
        fig_month = px.line(monthly_sales, x='month', y='sales', color='year', markers=True)
        st.plotly_chart(fig_month, width='stretch')

        st.subheader("Daily Average Sales (YoY)")
        daily_sales = filtered_df.groupby(['year', 'day_of_year'])['sales'].mean().reset_index()
        fig_day = px.line(daily_sales, x='day_of_year', y='sales', color='year')
        st.plotly_chart(fig_day, width='stretch')

        st.subheader("Average Sales by Day of the Month (YoY)")
        weekly_sales = filtered_df.groupby(['year', 'day_of_month'])['sales'].mean().reset_index()
        fig_week = px.line(weekly_sales, x='day_of_month', y='sales', color='year', markers=True)
        st.plotly_chart(fig_week, width='stretch')

        # average sales weekend vs weekday

        st.subheader("Average Sales by weekend vs weekday")
        weekend_sales = filtered_df.groupby(['year', 'is_weekend'])['sales'].mean().reset_index()
        fig_week = px.line(weekend_sales, x='is_weekend', y='sales', color='year', markers=True)
        st.plotly_chart(fig_week, width='stretch')


# ==========================================
# TAB 2: STORES & FAMILIES
# ==========================================
with tab2:
    st.header("Store and Product Analysis")
    
    st.subheader("Store Statistics")
    # Updated to use your new column names: store_type, store_city
    store_stats = filtered_df.groupby(['store_nbr', 'store_type', 'store_city']).agg(
        Total_Sales=('sales', 'sum'),
        Average_Sales=('sales', 'mean'),
        Max_Sales_in_a_day=('sales', 'max')
    ).reset_index().sort_values(by='Total_Sales', ascending=False)
    st.dataframe(store_stats, width='stretch')
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Store Type Over Time")
        type_sales = filtered_df.groupby(['year', 'store_type'])['sales'].mean().reset_index()
        fig_type = px.line(type_sales, x='year', y='sales', color='store_type', markers=True)
        st.plotly_chart(fig_type, width='stretch')

        st.subheader("Sales by Store Cluster Over Time")
        type_sales = filtered_df.groupby(['year', 'store_cluster'])['sales'].mean().reset_index()
        fig_type = px.line(type_sales, x='year', y='sales', color='store_cluster', markers=True)
        st.plotly_chart(fig_type, width='stretch')
        
        

    with col4:
        st.subheader("Top Product Families by Average Sales")
        family_sales = filtered_df.groupby('family')['sales'].mean().reset_index().sort_values('sales', ascending=True)
        fig_family = px.bar(family_sales.tail(20), x='sales', y='family', orientation='h')
        st.plotly_chart(fig_family, width='stretch')


        st.subheader("Total Sales Distribution by City")
        city_sales = filtered_df.groupby('store_city')['sales'].sum().reset_index()
        fig_city = px.pie(city_sales, values='sales', names='store_city', hole=0.4)
        st.plotly_chart(fig_city, width='stretch')

# ==========================================
# TAB 3: OIL & HOLIDAYS (New!)
# ==========================================
with tab3:
    st.header("Impact of External Factors")
    
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Oil Price vs Time")
        st.markdown("Ecuador's economy is highly dependent on oil. Do sales drop when oil drops?")
        # Group to daily level to prevent overplotting millions of rows
        daily_oil = df.groupby('date')['oil_price'].first().reset_index()
        fig_oil = px.line(daily_oil, x='date', y='oil_price')
        st.plotly_chart(fig_oil, width='stretch')
        
    with col8:
        st.subheader("Average Sales by Holiday Type")
        st.markdown("How much does holiday type affect purchasing?")
        holiday_sales = filtered_df.groupby('holiday_type')['sales'].mean().reset_index().sort_values('sales', ascending=False)
        fig_hol = px.bar(holiday_sales, x='holiday_type', y='sales', color='holiday_type')
        st.plotly_chart(fig_hol, width='stretch')

        st.subheader("Average Sales by Holiday Location")
        st.markdown("How much do local vs national holidays impact purchasing?")
        holiday_sales = filtered_df.groupby('holiday_location')['sales'].mean().reset_index().sort_values('sales', ascending=False)
        fig_hol = px.bar(holiday_sales, x='holiday_location', y='sales', color='holiday_location')
        st.plotly_chart(fig_hol, width='stretch')

        st.subheader("Average Sales by Holiday Locale")
        holiday_sales = filtered_df.groupby('holiday_location_name')['sales'].mean().reset_index().sort_values('sales', ascending=False)
        fig_hol = px.bar(holiday_sales, x='holiday_location_name', y='sales', color='holiday_location_name')
        st.plotly_chart(fig_hol, width='stretch')

# ==========================================
# TAB 4: AUTOCORRELATION (ACF/PACF)
# ==========================================
with tab4:
    st.header("Autocorrelation & Partial Autocorrelation")
    
    daily_total = filtered_df.groupby('date')['sales'].sum().reset_index()
    series = daily_total['sales'].values
    
    nlags = st.slider("Number of Lags", min_value=10, max_value=100, value=40, step=10)
    
    def plot_acf_pacf(series, nlags, plot_type='ACF'):
        if plot_type == 'ACF':
            corr_array, confint = acf(series, nlags=nlags, alpha=0.05)
        else:
            corr_array, confint = pacf(series, nlags=nlags, alpha=0.05)
            
        lower_bound = confint[:, 0] - corr_array
        upper_bound = confint[:, 1] - corr_array
        
        fig = go.Figure()
        for i in range(len(corr_array)):
            fig.add_shape(type='line', x0=i, y0=0, x1=i, y1=corr_array[i], line=dict(color='black', width=2))
        fig.add_trace(go.Scatter(x=list(range(len(corr_array))), y=corr_array, mode='markers', marker=dict(color='blue', size=8)))
        fig.add_trace(go.Scatter(x=list(range(len(corr_array))), y=upper_bound, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(len(corr_array))), y=lower_bound, mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        
        fig.update_layout(title=f"{plot_type} (95% Confidence Interval)", xaxis_title="Lag", yaxis_title="Correlation", showlegend=False)
        return fig

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_acf_pacf(series, nlags=nlags, plot_type='ACF'), width='stretch')
    with col6:
        st.plotly_chart(plot_acf_pacf(series, nlags=nlags, plot_type='PACF'), width='stretch')


# ==========================================
# TAB 5: PROMOTIONS (onpromotion)
# ==========================================
with tab5:
    st.header("Promotion Analysis")
    st.markdown("Analyze how the number of promoted items (`onpromotion`) impacts `sales`.")
    
    # Calculate the promotion-to-sales ratio safely (avoid dividing by zero)
    # If sales == 0, the ratio is set to 0 to prevent infinity/NaN errors.
    filtered_df['promo_to_sales_ratio'] = np.where(
        filtered_df['sales'] > 0, 
        filtered_df['onpromotion'] / filtered_df['sales'], 
        0
    )
    
    # --- KPI METRICS ROW ---
    st.subheader("High-Level Promotion Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # 1. Percentage of records with promotions
    pct_rows_promo = (filtered_df['onpromotion'] > 0).mean() * 100
    
    # 2. Percentage of TOTAL SALES driven by days with promotions
    sales_with_promo = filtered_df.loc[filtered_df['onpromotion'] > 0, 'sales'].sum()
    pct_sales_promo = (sales_with_promo / filtered_df['sales'].sum()) * 100 if filtered_df['sales'].sum() > 0 else 0
    
    # 3. Overall Correlation
    overall_corr = filtered_df['sales'].corr(filtered_df['onpromotion'])
    
    # 4. Variance
    var_nominal = filtered_df['onpromotion'].var()
    var_ratio = filtered_df['promo_to_sales_ratio'].var()
    
    metric_col1.metric("% Rows w/ Promotions", f"{pct_rows_promo:.2f}%")
    metric_col2.metric("% Total Sales w/ Promotions", f"{pct_sales_promo:.2f}%")
    metric_col3.metric("Sales vs Promo Correlation", f"{overall_corr:.3f}")
    metric_col4.metric("Variance (Nominal | Ratio)", f"{var_nominal:.1f} | {var_ratio:.3f}")
    
    st.divider()

    # --- CHARTS ---
    col9, col10 = st.columns(2)
    
    with col9:
        # Average (onpromotion/sales) over time
        st.subheader("Avg Promo-to-Sales Ratio Over Time")
        st.markdown("How many items are promoted per unit sold over time?")
        
        # Group by Year-Month for a smoother time series
        monthly_promo = filtered_df.groupby(['year', 'month']).agg(
            avg_ratio=('promo_to_sales_ratio', 'mean')
        ).reset_index()
        monthly_promo['date_str'] = monthly_promo['year'].astype(str) + '-' + monthly_promo['month'].astype(str).str.zfill(2)
        
        fig_ratio = px.line(monthly_promo, x='date_str', y='avg_ratio', markers=True,
                            labels={'date_str': 'Month', 'avg_ratio': 'Promo / Sales Ratio'})
        st.plotly_chart(fig_ratio, use_container_width=True)

        # Variance over time (Nominal)
        st.subheader("Variance of Promotions Over Time")
        monthly_var = filtered_df.groupby(['year', 'month'])['onpromotion'].var().reset_index()
        monthly_var['date_str'] = monthly_var['year'].astype(str) + '-' + monthly_var['month'].astype(str).str.zfill(2)
        
        fig_var = px.bar(monthly_var, x='date_str', y='onpromotion', 
                         labels={'date_str': 'Month', 'onpromotion': 'Variance of onpromotion'})
        st.plotly_chart(fig_var, use_container_width=True)

    with col10:
        # Correlation by Family
        st.subheader("Sales vs Promotions by Product Family")
        st.markdown("Which products are highly sensitive to promotions?")
        
        # Calculate correlation for each family independently
        family_corrs = []
        for fam in filtered_df['family'].unique():
            fam_df = filtered_df[filtered_df['family'] == fam]
            if len(fam_df) > 1 and fam_df['sales'].std() > 0 and fam_df['onpromotion'].std() > 0:
                corr = fam_df['sales'].corr(fam_df['onpromotion'])
                family_corrs.append({'family': fam, 'correlation': corr})
                
        if family_corrs:
            df_corrs = pd.DataFrame(family_corrs).sort_values('correlation', ascending=True)
            # Only show top/bottom to prevent overcrowding
            fig_fam_corr = px.bar(df_corrs, x='correlation', y='family', orientation='h',
                                  color='correlation', color_continuous_scale='RdBu')
            st.plotly_chart(fig_fam_corr, use_container_width=True)
        else:
            st.warning("Not enough variance in selected families to compute correlation.")

        # Scatter plot (Sales vs Promo aggregated by day to avoid plotting 3M dots)
        st.subheader("Daily Sales vs Daily Promotions")
        daily_scatter = filtered_df.groupby('date').agg(
            total_sales=('sales', 'sum'),
            total_promo=('onpromotion', 'sum')
        ).reset_index()
        
        fig_scatter = px.scatter(daily_scatter, x='total_promo', y='total_sales', 
                                 trendline="ols",
                                 labels={'total_promo': 'Total Daily Items on Promotion', 
                                         'total_sales': 'Total Daily Sales'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
# TAB 6: TRANSACTIONS (Foot Traffic)
# ==========================================
with tab6:
    st.header("Store Transactions (Foot Traffic YoY)")
    st.markdown("Analyze the total number of receipts/purchases. *Note: Test set dates will be blank.*")
    
    # Drop NaNs for transactions to avoid skewing the means (e.g., if merged with future test dates)
    tx_df = filtered_df.dropna(subset=['transactions'])
    
    col11, col12 = st.columns(2)
    
    with col11:
        st.subheader("Yearly Average Transactions")
        yearly_tx = tx_df.groupby('year')['transactions'].mean().reset_index()
        fig_year_tx = px.bar(yearly_tx, x='year', y='transactions', text_auto='.4s', color='year')
        st.plotly_chart(fig_year_tx, use_container_width=True)

        st.subheader("Weekly Average Transactions (YoY)")
        weekly_tx = tx_df.groupby(['year', 'week'])['transactions'].mean().reset_index()
        fig_week_tx = px.line(weekly_tx, x='week', y='transactions', color='year', markers=True)
        st.plotly_chart(fig_week_tx, use_container_width=True)

        st.subheader("Average Transactions by Day of the Week (YoY)")
        dow_tx = tx_df.groupby(['year', 'day_of_week'])['transactions'].mean().reset_index()
        fig_dow_tx = px.line(dow_tx, x='day_of_week', y='transactions', color='year', markers=True)
        # Update X-axis to show actual day names instead of 0-6
        fig_dow_tx.update_xaxes(tickmode='array', tickvals=[0,1,2,3,4,5,6], ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        st.plotly_chart(fig_dow_tx, use_container_width=True)

        st.subheader("Average Transactions: Payday vs Non-Payday")
        payday_tx = tx_df.groupby(['year', 'is_pay_day'])['transactions'].mean().reset_index()
        # Convert to string to treat as categorical for the X-axis
        payday_tx['is_pay_day'] = payday_tx['is_pay_day'].map({0: 'Non-Payday', 1: 'Payday'})
        fig_payday_tx = px.line(payday_tx, x='is_pay_day', y='transactions', color='year', markers=True)
        st.plotly_chart(fig_payday_tx, use_container_width=True)

    with col12:
        st.subheader("Monthly Average Transactions (YoY)")
        monthly_tx = tx_df.groupby(['year', 'month'])['transactions'].mean().reset_index()
        fig_month_tx = px.line(monthly_tx, x='month', y='transactions', color='year', markers=True)
        st.plotly_chart(fig_month_tx, use_container_width=True)

        st.subheader("Daily Average Transactions (YoY)")
        daily_tx = tx_df.groupby(['year', 'day_of_year'])['transactions'].mean().reset_index()
        fig_day_tx = px.line(daily_tx, x='day_of_year', y='transactions', color='year')
        st.plotly_chart(fig_day_tx, use_container_width=True)

        st.subheader("Average Transactions by Day of the Month (YoY)")
        dom_tx = tx_df.groupby(['year', 'day_of_month'])['transactions'].mean().reset_index()
        fig_dom_tx = px.line(dom_tx, x='day_of_month', y='transactions', color='year', markers=True)
        st.plotly_chart(fig_dom_tx, use_container_width=True)

        st.subheader("Average Transactions: Weekend vs Weekday")
        weekend_tx = tx_df.groupby(['year', 'is_weekend'])['transactions'].mean().reset_index()
        weekend_tx['is_weekend'] = weekend_tx['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        fig_weekend_tx = px.line(weekend_tx, x='is_weekend', y='transactions', color='year', markers=True)
        st.plotly_chart(fig_weekend_tx, use_container_width=True)



# other tabs
# correlation btwn on_promotion and sales - on_promotion vs family

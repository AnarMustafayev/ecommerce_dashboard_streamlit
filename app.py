import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Fancier Design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    .main {
        background-color: #F0F2F6;
    }

    /* KPI Card Styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }

    [data-testid="stMetricLabel"] {
        font-size: 16px;
        font-weight: 400;
        color: #4F4F4F !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: 700;
        color: #1E1E1E !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #F0F2F6;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #1a73e8;
        color: #1a73e8;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #e0e0e0;
    }

    /* Fix for sidebar text color in dark mode */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #1E1E1E;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-emotion-cache-1y4p8pa {
        color: #4F4F4F;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1629p8f, [data-testid="stSidebar"] .st-emotion-cache-q8sbsg {
        color: #1E1E1E; /* For selected values in widgets */
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    .stButton>button:hover {
        border-color: #ff6a6a;
        color: #ff6a6a;
    }

</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """
    Loads, merges, and preprocesses the e-commerce dataset.
    This function is cached to improve performance on subsequent runs.
    """
    try:
        customers = pd.read_csv('./olist_customers_dataset.csv')
        geolocation = pd.read_csv('./olist_geolocation_dataset.csv')
        orders = pd.read_csv('./olist_orders_dataset.csv')
        order_items = pd.read_csv('./olist_order_items_dataset.csv')
        products = pd.read_csv('./olist_products_dataset.csv')
        order_payments = pd.read_csv('./olist_order_payments_dataset.csv')
        order_reviews = pd.read_csv('./olist_order_reviews_dataset.csv')
        sellers = pd.read_csv('./olist_sellers_dataset.csv')
        category_translation = pd.read_csv('./product_category_name_translation.csv')
    except FileNotFoundError:
        st.error("One or more data files were not found. Please make sure all CSV files are in the same directory as the app.py file.")
        st.stop()

    df = orders.merge(customers, on='customer_id')
    df = df.merge(order_reviews, on='order_id')
    df = df.merge(order_payments, on='order_id')
    df = df.merge(order_items, on='order_id')
    df = df.merge(products, on='product_id')
    df = df.merge(sellers, on='seller_id')
    df = df.merge(category_translation, on='product_category_name')

    timestamp_cols = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date'
    ]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['delivery_delta'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    df['hour_of_day'] = df['order_purchase_timestamp'].dt.hour
    df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()

    def classify_delivery(delta):
        if pd.isna(delta): return 'Undefined'
        if delta >= 0: return 'On Time / Early'
        return 'Late'
    df['delivery_status'] = df['delivery_delta'].apply(classify_delivery)
    
    geo_df = geolocation.groupby('geolocation_state').agg({
        'geolocation_lat': 'mean', 'geolocation_lng': 'mean'
    }).reset_index()

    return df.dropna(subset=['order_purchase_timestamp']), geo_df

# --- Load Data ---
with st.spinner('Loading and processing data...'):
    df, geo_df = load_data()

# --- Initialize Session State for Filters ---
def initialize_session_state():
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = "All Years"
    if 'date_range' not in st.session_state:
        # Initialize as tuple of two datetime.date objects
        st.session_state.date_range = (
            df['order_purchase_timestamp'].min().date(), 
            df['order_purchase_timestamp'].max().date()
        )
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = "All States"
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "All Categories"

initialize_session_state()

def clear_filters():
    st.session_state.selected_year = "All Years"
    st.session_state.date_range = (df['order_purchase_timestamp'].min().date(), df['order_purchase_timestamp'].max().date())
    st.session_state.selected_state = "All States"
    st.session_state.selected_category = "All Categories"

# --- Sidebar Filters ---
st.sidebar.header("Filters")

with st.sidebar.expander("📅 Date Filters", expanded=True):
    all_years = sorted(df['purchase_year'].unique())
    selected_year = st.selectbox("Select Year:", ["All Years"] + all_years, key='selected_year')

    date_range_disabled = st.session_state.selected_year != "All Years"
    date_range = st.date_input(
        "Or Select Date Range:", value=st.session_state.date_range,
        min_value=df['order_purchase_timestamp'].min().date(),
        max_value=df['order_purchase_timestamp'].max().date(),
        disabled=date_range_disabled,
        key='date_range'
    )

with st.sidebar.expander("🌍 Location & Product Filters", expanded=True):
    all_states = sorted(df['customer_state'].unique())
    selected_state = st.selectbox("Select Customer State:", ["All States"] + all_states, key='selected_state')

    all_categories = sorted(df['product_category_name_english'].unique())
    selected_category = st.selectbox("Select Product Category:", ["All Categories"] + all_categories, key='selected_category')

st.sidebar.button("Clear All Filters", on_click=clear_filters, use_container_width=True)


# --- Apply Filters ---
filtered_df = df.copy()
if st.session_state.selected_year != "All Years":
    filtered_df = filtered_df[filtered_df['purchase_year'] == st.session_state.selected_year]
else:
    start_date, end_date = st.session_state.date_range
    filtered_df = filtered_df[
        (filtered_df['order_purchase_timestamp'].dt.date >= start_date) & 
        (filtered_df['order_purchase_timestamp'].dt.date <= end_date)
    ]

if st.session_state.selected_state != "All States":
    filtered_df = filtered_df[filtered_df['customer_state'] == st.session_state.selected_state]
if st.session_state.selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df['product_category_name_english'] == st.session_state.selected_category]


# --- Main Dashboard ---
st.title("E-commerce Dashboard")

# --- Key Metrics (KPIs) ---
if not filtered_df.empty:
    total_revenue = filtered_df['payment_value'].sum()
    total_orders = filtered_df['order_id'].nunique()
    avg_review_score = filtered_df['review_score'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="💰 Total Revenue (R$)", value=f"{total_revenue:,.2f}")
    with col2:
        st.metric(label="📦 Total Unique Orders", value=f"{total_orders:,}")
    with col3:
        st.metric(label="⭐ Average Review Score", value=f"{avg_review_score:.2f}")
else:
    st.warning("No data available for the selected filters. Please adjust your filter settings.")
    st.stop()

# --- Tabbed Layout ---
tab_sales, tab_customer, tab_product, tab_overview = st.tabs([ "📈 Sales Analysis", "👥 Customer Insights", "📦 Product & Seller","🌎 Overview"])

with tab_overview:
    st.subheader("Geographical Revenue Distribution")
    state_sales_geo = filtered_df.groupby('customer_state')['payment_value'].sum().reset_index()
    state_sales_map_data = state_sales_geo.merge(geo_df, left_on='customer_state', right_on='geolocation_state')
    
    fig_map = px.scatter_geo(
        state_sales_map_data,
        lat='geolocation_lat', lon='geolocation_lng', scope='south america',
        size='payment_value', hover_name='customer_state',
        hover_data={'payment_value': ':.2f', 'geolocation_lat': False, 'geolocation_lng': False},
        color='payment_value', color_continuous_scale='viridis_r'
    )
    fig_map.update_layout(
        geo=dict(center=dict(lon=-55, lat=-15), projection_scale=3.5, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_map, use_container_width=True)

# Continue with the rest of the tabs, such as `tab_sales`, `tab_customer`, and `tab_product`, as in your original code.

with tab_sales:
    st.subheader("Monthly Sales Revenue")
    monthly_sales = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['payment_value'].sum().reset_index()
    monthly_sales['order_purchase_timestamp'] = monthly_sales['order_purchase_timestamp'].dt.to_timestamp()
    fig_monthly_sales = px.line(
        monthly_sales, x='order_purchase_timestamp', y='payment_value', markers=True,
        labels={'payment_value': 'Total Revenue (R$)', 'order_purchase_timestamp': 'Month'}
    )
    fig_monthly_sales.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_monthly_sales, use_container_width=True)

    st.subheader("Weekly Sales Heatmap")
    sales_by_hour_day = filtered_df.groupby(['day_of_week', 'hour_of_day'])['payment_value'].sum().reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_pivot = sales_by_hour_day.pivot_table(index='day_of_week', columns='hour_of_day', values='payment_value', fill_value=0).reindex(weekday_order)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=sales_pivot.values, x=sales_pivot.columns, y=sales_pivot.index, colorscale='Blues',
        hovertemplate='<b>Day:</b> %{y}<br><b>Hour:</b> %{x}:00<br><b>Revenue:</b> R$ %{z:,.2f}<extra></extra>'
    ))
    fig_heatmap.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Day of Week",
        margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab_customer:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Payment Method Usage")
        payment_counts = filtered_df['payment_type'].value_counts().reset_index()
        payment_counts.columns = ['payment_type', 'count']
        fig_payment = px.pie(
            payment_counts, names='payment_type', values='count', hole=0.5,
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig_payment.update_traces(textposition='inside', textinfo='percent+label')
        fig_payment.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_payment, use_container_width=True)
    
    with col2:
        st.subheader("Delivery Promise Accuracy")
        delivery_status_counts = filtered_df['delivery_status'].value_counts().reset_index()
        delivery_status_counts.columns = ['status', 'count']
        fig_delivery = px.bar(
            delivery_status_counts, x='status', y='count', color='status',
            labels={'status': 'Delivery Status', 'count': 'Number of Orders'},
            color_discrete_map={'On Time / Early': '#2ca02c', 'Late': '#d62728', 'Undefined': '#7f7f7f'}
        )
        fig_delivery.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_delivery, use_container_width=True)

    st.subheader("Impact of Delivery Time on Customer Reviews")
    fig_delivery_score = px.box(
        filtered_df, x='review_score', y='delivery_time',
        labels={'review_score': 'Review Score', 'delivery_time': 'Delivery Time (Days)'},
        color_discrete_sequence=px.colors.sequential.PuBu
    )
    fig_delivery_score.update_layout(yaxis_range=[0, 60], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_delivery_score, use_container_width=True)

with tab_product:
    st.subheader("Top Product & Seller Performance")
    top_n = st.slider("Select number of top items to display:", 5, 20, 10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### By Product Category")
        category_sales = filtered_df.groupby('product_category_name_english')['payment_value'].sum().sort_values(ascending=False).head(top_n).reset_index()
        fig_top_cat = px.bar(
            category_sales, x='payment_value', y='product_category_name_english', orientation='h',
            labels={'payment_value': 'Total Revenue (R$)', 'product_category_name_english': 'Product Category'},
            color='payment_value', color_continuous_scale=px.colors.sequential.Purples
        )
        fig_top_cat.update_layout(
            yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_top_cat, use_container_width=True)

    with col2:
        st.markdown("##### By Seller State")
        seller_revenue = filtered_df.groupby('seller_state')['payment_value'].sum().sort_values(ascending=False).head(top_n)
        fig_seller_state = px.bar(
            seller_revenue, x=seller_revenue.values, y=seller_revenue.index, orientation='h',
            labels={'x': 'Total Revenue (R$)', 'y': 'Seller State'},
            color=seller_revenue.values, color_continuous_scale=px.colors.sequential.Aggrnyl
        )
        fig_seller_state.update_layout(
            yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_seller_state, use_container_width=True)

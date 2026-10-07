"""
===============================================================================
FOOD DELIVERY BUSINESS INTELLIGENCE & AI-BASED CUSTOMER RETENTION DASHBOARD
Turning Food Delivery Data into Actionable Business Decisions
===============================================================================

Dataset: FoodHub Order Analysis Dataset
Source:   https://github.com/prneidhardt/Python-Foundations (Public GitHub)
Records: ~1,898 food delivery orders from a NYC-based food aggregator
Columns: order_id, customer_id, restaurant_name, cuisine_type,
         cost_of_the_order, day_of_the_week, rating, food_preparation_time,
         delivery_time

Author:  Rajiv (Business Intelligence Internship Project)
Tech:    Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FoodHub BI Dashboard",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL UI
# ============================================================================
st.markdown("""
<style>
    /* Main page styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* KPI Card Styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Insight Card Styling */
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 15px 20px;
        margin: 10px 0;
    }
    .insight-card-warning {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        border-radius: 0 8px 8px 0;
        padding: 15px 20px;
        margin: 10px 0;
    }
    .insight-card-success {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        border-radius: 0 8px 8px 0;
        padding: 15px 20px;
        margin: 10px 0;
    }
    .insight-card-danger {
        background: #ffebee;
        border-left: 4px solid #f44336;
        border-radius: 0 8px 8px 0;
        padding: 15px 20px;
        margin: 10px 0;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Risk badges */
    .risk-high { color: #f44336; font-weight: bold; }
    .risk-medium { color: #ff9800; font-weight: bold; }
    .risk-low { color: #4caf50; font-weight: bold; }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric delta styling */
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
@st.cache_data
def load_data():
    """
    Load the FoodHub dataset from local file or remote source.
    Returns the raw dataframe and metadata about the loading process.
    """
    data_info = {
        'source': 'FoodHub Order Analysis Dataset',
        'url': 'https://raw.githubusercontent.com/prneidhardt/Python-Foundations/main/foodhub_order.csv',
        'local_path': 'foodhub_order.csv',
        'load_method': None,
        'original_rows': 0,
        'original_cols': 0
    }

    df = None

    # Try local file first
    local_paths = ['foodhub_order.csv', 'data/foodhub_order.csv']
    for path in local_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                data_info['load_method'] = f'Local file: {path}'
                break
            except Exception:
                continue

    # Try remote URL if local not found
    if df is None:
        try:
            df = pd.read_csv(data_info['url'])
            data_info['load_method'] = 'Remote URL (GitHub)'
        except Exception:
            st.error("❌ Could not load dataset. Please ensure 'foodhub_order.csv' "
                     "is in the project directory.")
            st.info(f"Download from: {data_info['url']}")
            st.stop()

    data_info['original_rows'] = len(df)
    data_info['original_cols'] = len(df.columns)

    return df, data_info


# ============================================================================
# DATA CLEANING FUNCTION
# ============================================================================
@st.cache_data
def clean_data(df):
    """
    Clean the raw dataset:
    - Handle missing values
    - Convert data types
    - Handle 'Not given' ratings
    - Remove duplicates
    - Detect and handle outliers
    Returns cleaned dataframe and cleaning report.
    """
    cleaning_report = {
        'original_rows': len(df),
        'duplicates_found': 0,
        'duplicates_removed': 0,
        'rating_not_given_count': 0,
        'missing_values': {},
        'outliers_handled': 0,
        'invalid_records': 0,
        'final_rows': 0
    }

    df_clean = df.copy()

    # --- Check for missing values ---
    cleaning_report['missing_values'] = df_clean.isnull().sum().to_dict()

    # --- Handle duplicates ---
    dups = df_clean.duplicated().sum()
    cleaning_report['duplicates_found'] = dups
    if dups > 0:
        df_clean = df_clean.drop_duplicates()
        cleaning_report['duplicates_removed'] = dups

    # --- Handle 'Not given' ratings ---
    # Count ratings that are 'Not given'
    if 'rating' in df_clean.columns:
        not_given_mask = df_clean['rating'].astype(str).str.strip().str.lower() == 'not given'
        cleaning_report['rating_not_given_count'] = not_given_mask.sum()

        # Create a numeric rating column (NaN for 'Not given')
        df_clean['rating_numeric'] = pd.to_numeric(
            df_clean['rating'].replace('Not given', np.nan), errors='coerce'
        )

        # Create a flag for whether rating was given
        df_clean['rating_given'] = ~not_given_mask

    # --- Ensure correct data types ---
    numeric_cols = ['cost_of_the_order', 'food_preparation_time', 'delivery_time']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # --- Handle negative/zero values in cost ---
    if 'cost_of_the_order' in df_clean.columns:
        invalid_cost = (df_clean['cost_of_the_order'] <= 0).sum()
        cleaning_report['invalid_records'] += invalid_cost
        df_clean = df_clean[df_clean['cost_of_the_order'] > 0]

    # --- Handle outliers using IQR method (flag, don't remove) ---
    if 'cost_of_the_order' in df_clean.columns:
        Q1 = df_clean['cost_of_the_order'].quantile(0.25)
        Q3 = df_clean['cost_of_the_order'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean['cost_outlier'] = (
            (df_clean['cost_of_the_order'] < lower) |
            (df_clean['cost_of_the_order'] > upper)
        )
        cleaning_report['outliers_handled'] = df_clean['cost_outlier'].sum()

    # --- Create total time column ---
    if 'food_preparation_time' in df_clean.columns and 'delivery_time' in df_clean.columns:
        df_clean['total_time'] = (
            df_clean['food_preparation_time'] + df_clean['delivery_time']
        )

    cleaning_report['final_rows'] = len(df_clean)

    return df_clean, cleaning_report


# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================
@st.cache_data
def engineer_features(df):
    """
    Create new features for analysis and ML:
    - Order value categories
    - Delivery speed categories
    - Customer-level aggregations
    - Revenue contribution metrics
    """
    df_feat = df.copy()

    # --- Order Value Categories ---
    if 'cost_of_the_order' in df_feat.columns:
        df_feat['order_value_category'] = pd.cut(
            df_feat['cost_of_the_order'],
            bins=[0, 10, 20, 30, 50, float('inf')],
            labels=['Budget (<$10)', 'Economy ($10-20)',
                    'Standard ($20-30)', 'Premium ($30-50)', 'Luxury ($50+)']
        )

    # --- Delivery Speed Categories ---
    if 'delivery_time' in df_feat.columns:
        df_feat['delivery_speed'] = pd.cut(
            df_feat['delivery_time'],
            bins=[0, 15, 25, 35, float('inf')],
            labels=['Fast (<15min)', 'Normal (15-25min)',
                    'Slow (25-35min)', 'Very Slow (>35min)']
        )

    # --- Preparation Time Categories ---
    if 'food_preparation_time' in df_feat.columns:
        df_feat['prep_speed'] = pd.cut(
            df_feat['food_preparation_time'],
            bins=[0, 20, 27, 35, float('inf')],
            labels=['Quick (<20min)', 'Average (20-27min)',
                    'Slow (27-35min)', 'Very Slow (>35min)']
        )

    # --- Weekend/Weekday binary ---
    if 'day_of_the_week' in df_feat.columns:
        df_feat['is_weekend'] = (
            df_feat['day_of_the_week'].str.strip().str.lower() == 'weekend'
        ).astype(int)

    # --- Rating category ---
    if 'rating_numeric' in df_feat.columns:
        conditions = [
            df_feat['rating_numeric'] >= 4,
            df_feat['rating_numeric'] >= 3,
            df_feat['rating_numeric'] >= 1
        ]
        choices = ['High (4-5)', 'Medium (3)', 'Low (1-2)']
        df_feat['rating_category'] = np.select(
            conditions, choices, default='Not Rated'
        )

    return df_feat


# ============================================================================
# KPI CALCULATION FUNCTION
# ============================================================================
@st.cache_data
def calculate_kpis(df):
    """
    Calculate all key performance indicators from the dataset.
    Only computes KPIs for columns that actually exist.
    """
    kpis = {}

    # --- Revenue/Order KPIs ---
    if 'cost_of_the_order' in df.columns:
        kpis['total_revenue'] = df['cost_of_the_order'].sum()
        kpis['avg_order_value'] = df['cost_of_the_order'].mean()
        kpis['median_order_value'] = df['cost_of_the_order'].median()
        kpis['max_order_value'] = df['cost_of_the_order'].max()
        kpis['min_order_value'] = df['cost_of_the_order'].min()

    kpis['total_orders'] = len(df)

    # --- Customer KPIs ---
    if 'customer_id' in df.columns:
        kpis['total_customers'] = df['customer_id'].nunique()
        orders_per_customer = df.groupby('customer_id').size()
        kpis['avg_orders_per_customer'] = orders_per_customer.mean()
        kpis['repeat_customers'] = (orders_per_customer > 1).sum()
        kpis['repeat_rate'] = (
            kpis['repeat_customers'] / kpis['total_customers'] * 100
            if kpis['total_customers'] > 0 else 0
        )
        if 'cost_of_the_order' in df.columns:
            kpis['revenue_per_customer'] = (
                kpis['total_revenue'] / kpis['total_customers']
            )

    # --- Restaurant KPIs ---
    if 'restaurant_name' in df.columns:
        kpis['total_restaurants'] = df['restaurant_name'].nunique()

    # --- Rating KPIs ---
    if 'rating_numeric' in df.columns:
        rated = df['rating_numeric'].dropna()
        kpis['avg_rating'] = rated.mean() if len(rated) > 0 else 0
        kpis['rating_given_pct'] = len(rated) / len(df) * 100

    # --- Delivery KPIs ---
    if 'delivery_time' in df.columns:
        kpis['avg_delivery_time'] = df['delivery_time'].mean()
        kpis['median_delivery_time'] = df['delivery_time'].median()

    if 'food_preparation_time' in df.columns:
        kpis['avg_prep_time'] = df['food_preparation_time'].mean()

    if 'total_time' in df.columns:
        kpis['avg_total_time'] = df['total_time'].mean()

    # --- Cuisine KPIs ---
    if 'cuisine_type' in df.columns:
        kpis['total_cuisines'] = df['cuisine_type'].nunique()
        kpis['top_cuisine'] = df['cuisine_type'].value_counts().index[0]

    # --- Day KPIs ---
    if 'day_of_the_week' in df.columns:
        day_counts = df['day_of_the_week'].value_counts()
        kpis['weekend_orders'] = day_counts.get('Weekend', 0)
        kpis['weekday_orders'] = day_counts.get('Weekday', 0)
        kpis['weekend_pct'] = (
            kpis['weekend_orders'] / kpis['total_orders'] * 100
        )

    return kpis


# ============================================================================
# CUSTOMER ANALYSIS FUNCTION
# ============================================================================
@st.cache_data
def perform_customer_analysis(df):
    """
    Perform comprehensive customer-level analysis:
    - RFM-style segmentation (adapted for available data)
    - Customer value tiers
    - Behavioral patterns
    """
    if 'customer_id' not in df.columns:
        return None, None

    # --- Build customer-level features ---
    customer_df = df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_spend=('cost_of_the_order', 'sum'),
        avg_order_value=('cost_of_the_order', 'mean'),
        max_order_value=('cost_of_the_order', 'max'),
        avg_delivery_time=('delivery_time', 'mean'),
        avg_prep_time=('food_preparation_time', 'mean'),
        cuisines_tried=('cuisine_type', 'nunique'),
        restaurants_tried=('restaurant_name', 'nunique'),
    ).reset_index()

    # Add rating metrics
    if 'rating_numeric' in df.columns:
        rating_agg = df.groupby('customer_id').agg(
            avg_rating=('rating_numeric', 'mean'),
            ratings_given=('rating_given', 'sum'),
        ).reset_index()
        customer_df = customer_df.merge(rating_agg, on='customer_id', how='left')
        customer_df['rating_response_rate'] = (
            customer_df['ratings_given'] / customer_df['total_orders'] * 100
        )

    # Add weekend preference
    if 'is_weekend' in df.columns:
        weekend_agg = df.groupby('customer_id')['is_weekend'].mean().reset_index()
        weekend_agg.columns = ['customer_id', 'weekend_order_pct']
        weekend_agg['weekend_order_pct'] *= 100
        customer_df = customer_df.merge(weekend_agg, on='customer_id', how='left')

    # --- RFM-Style Segmentation ---
    # Since we don't have date data, we adapt RFM:
    # F = Frequency (total_orders)
    # M = Monetary (total_spend)
    # E = Engagement (rating_response_rate + cuisines_tried diversity)

    # Frequency Score (1-5)
    customer_df['frequency_score'] = pd.qcut(
        customer_df['total_orders'].rank(method='first'),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Monetary Score (1-5)
    customer_df['monetary_score'] = pd.qcut(
        customer_df['total_spend'].rank(method='first'),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Engagement Score (based on available metrics)
    engagement_factors = []
    if 'rating_response_rate' in customer_df.columns:
        engagement_factors.append(
            pd.qcut(
                customer_df['rating_response_rate'].rank(method='first'),
                q=5, labels=[1, 2, 3, 4, 5]
            ).astype(int)
        )
    engagement_factors.append(
        pd.qcut(
            customer_df['cuisines_tried'].rank(method='first'),
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    )

    customer_df['engagement_score'] = sum(engagement_factors) / len(engagement_factors)
    customer_df['engagement_score'] = customer_df['engagement_score'].round().astype(int)

    # Composite Score
    customer_df['composite_score'] = (
        customer_df['frequency_score'] +
        customer_df['monetary_score'] +
        customer_df['engagement_score']
    )

    # --- Segment Assignment ---
    def assign_segment(row):
        f, m, e = row['frequency_score'], row['monetary_score'], row['engagement_score']
        composite = row['composite_score']

        if f >= 4 and m >= 4:
            return 'Champions'
        elif f >= 3 and m >= 3 and e >= 3:
            return 'Loyal Customers'
        elif (f >= 3 and m >= 2) or (m >= 3 and e >= 3):
            return 'Potential Loyalists'
        elif f <= 2 and m >= 3:
            return 'At Risk'
        elif f <= 2 and m <= 2 and e <= 2:
            return 'Lost / Inactive'
        elif f <= 2 and m <= 2:
            return 'Need Attention'
        else:
            return 'New / Occasional'

    customer_df['segment'] = customer_df.apply(assign_segment, axis=1)

    # --- Customer Value Tier ---
    customer_df['value_tier'] = pd.qcut(
        customer_df['total_spend'].rank(method='first'),
        q=4, labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
    )

    return customer_df, df


# ============================================================================
# AI/ML RISK SCORING FUNCTION
# ============================================================================
@st.cache_data
def train_risk_model(customer_df):
    """
    Train an AI-Assisted Customer Risk Scoring model.

    Since the dataset does not contain temporal/date information for true churn
    prediction, we implement a behavioral risk scoring system using available
    features: frequency, monetary value, engagement, and service experience.

    We use a supervised approach where 'at-risk' customers are defined by
    low frequency + low engagement + poor service experience indicators.
    """
    if customer_df is None or len(customer_df) < 50:
        return None, None, None

    model_df = customer_df.copy()

    # --- Define Risk Label ---
    # A customer is labeled as 'at-risk' based on behavioral signals:
    # - Low order frequency (bottom 30%)
    # - Low engagement (few ratings given, limited cuisine exploration)
    # - Poor service experience (high delivery times, low ratings)

    freq_threshold = model_df['total_orders'].quantile(0.30)
    spend_threshold = model_df['total_spend'].quantile(0.30)

    risk_score = np.zeros(len(model_df))

    # Low frequency indicator
    risk_score += (model_df['total_orders'] <= freq_threshold).astype(int) * 2

    # Low spend indicator
    risk_score += (model_df['total_spend'] <= spend_threshold).astype(int) * 2

    # Low engagement (few ratings given)
    if 'rating_response_rate' in model_df.columns:
        risk_score += (model_df['rating_response_rate'] < 30).astype(int)

    # Poor service experience
    avg_del = model_df['avg_delivery_time'].mean()
    risk_score += (model_df['avg_delivery_time'] > avg_del * 1.2).astype(int)

    # Low cuisine diversity
    risk_score += (model_df['cuisines_tried'] <= 1).astype(int)

    # Binary label: at-risk if risk_score >= 4
    model_df['at_risk'] = (risk_score >= 4).astype(int)

    # --- Prepare Features ---
    feature_cols = ['total_orders', 'total_spend', 'avg_order_value',
                    'avg_delivery_time', 'avg_prep_time',
                    'cuisines_tried', 'restaurants_tried']

    if 'rating_response_rate' in model_df.columns:
        feature_cols.append('rating_response_rate')
    if 'avg_rating' in model_df.columns:
        # Fill NaN ratings with median
        model_df['avg_rating'] = model_df['avg_rating'].fillna(
            model_df['avg_rating'].median()
        )
        feature_cols.append('avg_rating')
    if 'weekend_order_pct' in model_df.columns:
        feature_cols.append('weekend_order_pct')

    X = model_df[feature_cols].fillna(0)
    y = model_df['at_risk']

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Scale Features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Multiple Models ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=5
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=3
        )
    }

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
        }
        results[name] = metrics

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = (name, model)

    # --- Get Feature Importance from best model ---
    model_name, trained_model = best_model
    if hasattr(trained_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': trained_model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(trained_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(trained_model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': 0})

    # --- Predict risk scores for all customers ---
    X_all = model_df[feature_cols].fillna(0)
    if model_name == 'Logistic Regression':
        X_all_scaled = scaler.transform(X_all)
        risk_probabilities = trained_model.predict_proba(X_all_scaled)[:, 1]
    else:
        risk_probabilities = trained_model.predict_proba(X_all)[:, 1]

    model_df['risk_probability'] = risk_probabilities

    # Assign risk categories
    model_df['risk_category'] = pd.cut(
        model_df['risk_probability'],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    model_output = {
        'model_name': model_name,
        'results': results,
        'feature_importance': feature_importance,
        'best_metrics': results[model_name],
        'customer_risk': model_df[['customer_id', 'risk_probability', 'risk_category',
                                    'total_orders', 'total_spend', 'segment']],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'risk_distribution': model_df['risk_category'].value_counts().to_dict(),
        'at_risk_pct': model_df['at_risk'].mean() * 100,
        'feature_cols': feature_cols
    }

    return model_output, model_df, results


# ============================================================================
# BUSINESS INSIGHTS GENERATOR
# ============================================================================
def generate_business_insights(df, kpis, customer_df=None):
    """
    Automatically generate business insights from actual data.
    Each insight is traceable to a calculated metric.
    """
    insights = []

    # --- Revenue Driver Insight ---
    if 'cuisine_type' in df.columns and 'cost_of_the_order' in df.columns:
        cuisine_rev = df.groupby('cuisine_type')['cost_of_the_order'].sum()
        top_cuisine = cuisine_rev.idxmax()
        top_cuisine_pct = cuisine_rev.max() / cuisine_rev.sum() * 100
        bottom_cuisine = cuisine_rev.idxmin()
        bottom_cuisine_pct = cuisine_rev.min() / cuisine_rev.sum() * 100

        insights.append({
            'type': 'revenue_driver',
            'icon': '💰',
            'fact': f"{top_cuisine} cuisine generates ${cuisine_rev.max():,.2f} "
                    f"({top_cuisine_pct:.1f}% of total revenue).",
            'insight': f"{top_cuisine} is the dominant revenue driver, while "
                       f"{bottom_cuisine} contributes only {bottom_cuisine_pct:.1f}%.",
            'meaning': f"Business is heavily reliant on {top_cuisine} cuisine demand. "
                       f"Diversification could reduce concentration risk.",
            'category': 'strength'
        })

    # --- Weekend vs Weekday Pattern ---
    if 'weekend_pct' in kpis:
        if kpis['weekend_pct'] > 55:
            insights.append({
                'type': 'demand_pattern',
                'icon': '📅',
                'fact': f"Weekend orders comprise {kpis['weekend_pct']:.1f}% of total orders.",
                'insight': "Demand skews heavily toward weekends, suggesting "
                           "leisure/occasion-driven ordering.",
                'meaning': "Weekday demand represents a growth opportunity. "
                           "Targeted weekday promotions could boost volumes.",
                'category': 'opportunity'
            })
        else:
            insights.append({
                'type': 'demand_pattern',
                'icon': '📅',
                'fact': f"Weekend orders comprise {kpis['weekend_pct']:.1f}%, "
                        f"Weekday orders are {100 - kpis['weekend_pct']:.1f}%.",
                'insight': "Demand is relatively balanced across the week.",
                'meaning': "The platform has consistent daily engagement, "
                           "indicating established habitual ordering behavior.",
                'category': 'strength'
            })

    # --- Rating Insight ---
    if 'avg_rating' in kpis and 'rating_given_pct' in kpis:
        insights.append({
            'type': 'customer_satisfaction',
            'icon': '⭐',
            'fact': f"Average rating is {kpis['avg_rating']:.2f}/5, but only "
                    f"{kpis['rating_given_pct']:.1f}% of orders received a rating.",
            'insight': f"The {100 - kpis['rating_given_pct']:.1f}% unrated orders "
                       f"represent a feedback gap that obscures true satisfaction levels.",
            'meaning': "Improving rating collection is critical for understanding "
                       "customer satisfaction and identifying service issues.",
            'category': 'risk'
        })

    # --- Delivery Performance ---
    if 'avg_delivery_time' in kpis and 'avg_prep_time' in kpis:
        total = kpis.get('avg_total_time', kpis['avg_delivery_time'] + kpis['avg_prep_time'])
        prep_pct = kpis['avg_prep_time'] / total * 100

        insights.append({
            'type': 'operations',
            'icon': '🚚',
            'fact': f"Average total order time is {total:.0f} minutes "
                    f"(Prep: {kpis['avg_prep_time']:.0f}min + "
                    f"Delivery: {kpis['avg_delivery_time']:.0f}min).",
            'insight': f"Food preparation accounts for {prep_pct:.0f}% of total "
                       f"wait time, making it the primary operational bottleneck.",
            'meaning': "Optimizing restaurant preparation workflows could "
                       "significantly reduce overall order completion time.",
            'category': 'risk' if total > 50 else 'neutral'
        })

    # --- Repeat Customer Insight ---
    if 'repeat_rate' in kpis:
        insights.append({
            'type': 'retention',
            'icon': '🔄',
            'fact': f"Repeat customer rate is {kpis['repeat_rate']:.1f}% "
                    f"({kpis.get('repeat_customers', 0):,} out of "
                    f"{kpis.get('total_customers', 0):,} customers).",
            'insight': "Each percentage point improvement in repeat rate directly "
                       "impacts customer lifetime value and reduces acquisition cost.",
            'meaning': "Retention programs targeting single-order customers could "
                       "unlock significant revenue growth.",
            'category': 'opportunity' if kpis['repeat_rate'] < 30 else 'strength'
        })

    # --- Restaurant Concentration ---
    if 'restaurant_name' in df.columns and 'cost_of_the_order' in df.columns:
        rest_rev = df.groupby('restaurant_name')['cost_of_the_order'].sum().sort_values(
            ascending=False
        )
        top5_rev = rest_rev.head(5).sum()
        top5_pct = top5_rev / rest_rev.sum() * 100

        insights.append({
            'type': 'concentration',
            'icon': '🏪',
            'fact': f"Top 5 restaurants generate ${top5_rev:,.2f} "
                    f"({top5_pct:.1f}% of total revenue).",
            'insight': f"Revenue is {'highly concentrated' if top5_pct > 30 else 'well distributed'} "
                       f"among the top restaurants.",
            'meaning': "High concentration means the business is vulnerable to "
                       "losing key restaurant partners."
                       if top5_pct > 30 else
                       "Revenue diversification reduces partner dependency risk.",
            'category': 'risk' if top5_pct > 30 else 'strength'
        })

    # --- Customer Segment Insight (if available) ---
    if customer_df is not None and 'segment' in customer_df.columns:
        seg_counts = customer_df['segment'].value_counts()
        champions_pct = seg_counts.get('Champions', 0) / len(customer_df) * 100
        at_risk_pct = seg_counts.get('At Risk', 0) / len(customer_df) * 100

        if champions_pct > 0:
            champ_rev = customer_df[customer_df['segment'] == 'Champions']['total_spend'].sum()
            insights.append({
                'type': 'customer_segment',
                'icon': '🏆',
                'fact': f"Champions ({champions_pct:.1f}% of customers) "
                        f"generate ${champ_rev:,.2f} in revenue.",
                'insight': "These high-frequency, high-value customers are the "
                           "backbone of the business.",
                'meaning': "Protecting Champions with loyalty rewards and VIP "
                           "treatment should be the top retention priority.",
                'category': 'strength'
            })

        if at_risk_pct > 0:
            at_risk_rev = customer_df[customer_df['segment'] == 'At Risk']['total_spend'].sum()
            insights.append({
                'type': 'customer_risk',
                'icon': '⚠️',
                'fact': f"At-Risk customers ({at_risk_pct:.1f}%) have "
                        f"${at_risk_rev:,.2f} in historical spend at stake.",
                'insight': "These previously active customers show declining engagement.",
                'meaning': "A targeted win-back campaign could recover significant "
                           "at-risk revenue.",
                'category': 'risk'
            })

    return insights


# ============================================================================
# RECOMMENDATIONS GENERATOR
# ============================================================================
def generate_recommendations(df, kpis, customer_df=None, model_output=None):
    """
    Generate actionable business recommendations from analysis results.
    Each recommendation follows: FACT → INSIGHT → RISK/OPPORTUNITY → ACTION
    """
    recommendations = []

    # --- Recommendation 1: Rating Collection Improvement ---
    if 'rating_given_pct' in kpis and kpis['rating_given_pct'] < 80:
        unrated_pct = 100 - kpis['rating_given_pct']
        recommendations.append({
            'title': 'Improve Customer Feedback Collection',
            'fact': f"{unrated_pct:.1f}% of orders have no rating provided.",
            'insight': "Low feedback rates create blind spots in service quality monitoring.",
            'risk_opportunity': 'Operational Risk — inability to detect quality issues early',
            'action': "Implement in-app rating prompts with incentives (e.g., "
                      "small discount on next order). Set a target of 70%+ rating collection.",
            'impact': 'High',
            'urgency': 'High',
            'category': 'Operations'
        })

    # --- Recommendation 2: Weekend/Weekday Balancing ---
    if 'weekend_pct' in kpis and kpis['weekend_pct'] > 55:
        recommendations.append({
            'title': 'Boost Weekday Order Volume',
            'fact': f"Weekend orders ({kpis['weekend_pct']:.1f}%) significantly "
                    f"outpace weekday orders.",
            'insight': "Underutilized weekday capacity represents lost revenue potential.",
            'risk_opportunity': 'Growth Opportunity — weekday promotions could lift total volume by 15-25%',
            'action': "Launch 'Weekday Specials' with 10-15% discounts on select cuisines. "
                      "Partner with offices for corporate lunch programs.",
            'impact': 'High',
            'urgency': 'Medium',
            'category': 'Revenue Growth'
        })

    # --- Recommendation 3: Delivery Time Optimization ---
    if 'avg_delivery_time' in kpis and kpis['avg_delivery_time'] > 24:
        recommendations.append({
            'title': 'Reduce Delivery Wait Times',
            'fact': f"Average delivery time is {kpis['avg_delivery_time']:.0f} minutes.",
            'insight': "Long delivery times correlate with lower customer satisfaction and repeat rates.",
            'risk_opportunity': 'Retention Risk — customers may switch to faster competitors',
            'action': "Optimize delivery routing, add delivery partners in high-demand zones, "
                      "and implement real-time ETAs to manage expectations.",
            'impact': 'High',
            'urgency': 'High',
            'category': 'Operations'
        })

    # --- Recommendation 4: Customer Retention Program ---
    if customer_df is not None and 'segment' in customer_df.columns:
        one_time = customer_df[customer_df['total_orders'] == 1]
        if len(one_time) > 0:
            one_time_pct = len(one_time) / len(customer_df) * 100
            one_time_rev = one_time['total_spend'].sum()
            recommendations.append({
                'title': 'Convert One-Time Buyers to Repeat Customers',
                'fact': f"{one_time_pct:.1f}% of customers ({len(one_time):,}) "
                        f"placed only one order (${one_time_rev:,.2f} total).",
                'insight': "Single-order customers represent the largest untapped retention pool.",
                'risk_opportunity': 'Revenue Opportunity — converting even 10% to repeat '
                                    'buyers could add significant revenue',
                'action': "Send personalized re-engagement emails within 7 days of first order. "
                          "Offer a 'Second Order Discount' (15-20% off). "
                          "Show cuisine recommendations based on first order.",
                'impact': 'High',
                'urgency': 'Medium',
                'category': 'Customer Retention'
            })

    # --- Recommendation 5: Top Restaurant Partnerships ---
    if 'restaurant_name' in df.columns and 'cost_of_the_order' in df.columns:
        rest_stats = df.groupby('restaurant_name').agg(
            orders=('order_id', 'count'),
            revenue=('cost_of_the_order', 'sum')
        ).sort_values('revenue', ascending=False)

        top_rest = rest_stats.head(1)
        recommendations.append({
            'title': 'Strengthen Top Restaurant Partnerships',
            'fact': f"{top_rest.index[0]} leads with {top_rest['orders'].values[0]} orders "
                    f"and ${top_rest['revenue'].values[0]:,.2f} revenue.",
            'insight': "Top-performing restaurants are critical to platform success and user retention.",
            'risk_opportunity': 'Strategic Risk — losing a top restaurant partner could '
                                'significantly impact revenue',
            'action': "Negotiate exclusive deals with top 10 restaurants. "
                      "Offer premium placement in search results. "
                      "Establish quarterly business reviews with key partners.",
            'impact': 'High',
            'urgency': 'Low',
            'category': 'Partnerships'
        })

    # --- Recommendation 6: Cuisine Diversification ---
    if 'cuisine_type' in df.columns:
        cuisine_orders = df['cuisine_type'].value_counts()
        bottom_cuisines = cuisine_orders.tail(2)
        if len(bottom_cuisines) > 0:
            recommendations.append({
                'title': 'Promote Underperforming Cuisines',
                'fact': f"{', '.join(bottom_cuisines.index.tolist())} cuisines have the "
                        f"lowest order volumes ({', '.join(str(v) for v in bottom_cuisines.values)} orders).",
                'insight': "Low-performing cuisines may indicate either lack of demand or "
                           "insufficient visibility on the platform.",
                'risk_opportunity': 'Growth Opportunity — targeted promotion could unlock hidden demand',
                'action': "Feature underperforming cuisines in 'Discover' sections. "
                          "Run 'Try Something New' campaigns with introductory discounts. "
                          "Add curated collections and food stories.",
                'impact': 'Medium',
                'urgency': 'Low',
                'category': 'Revenue Growth'
            })

    # --- Recommendation 7: AI-Driven Risk Mitigation ---
    if model_output is not None:
        high_risk_count = model_output['risk_distribution'].get('High Risk', 0)
        if high_risk_count > 0:
            recommendations.append({
                'title': 'Implement AI-Driven Customer Retention',
                'fact': f"{high_risk_count} customers identified as High Risk by "
                        f"the ML model ({model_output['model_name']}).",
                'insight': "Predictive analytics can identify at-risk customers before they churn.",
                'risk_opportunity': 'Revenue Protection — proactive intervention is 5x cheaper than re-acquisition',
                'action': "Deploy automated alerts for high-risk customers. "
                          "Design personalized win-back campaigns based on risk factors. "
                          "Monitor model performance monthly and retrain quarterly.",
                'impact': 'High',
                'urgency': 'High',
                'category': 'Customer Retention'
            })

    return recommendations


# ============================================================================
# HELPER: KPI CARD RENDERER
# ============================================================================
def render_kpi_card(label, value, card_class="kpi-card"):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="{card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HELPER: INSIGHT CARD RENDERER
# ============================================================================
def render_insight_card(icon, fact, insight, meaning, card_type="insight-card"):
    """Render a styled business insight card."""
    st.markdown(f"""
    <div class="{card_type}">
        <strong>{icon} FACT:</strong> {fact}<br><br>
        <strong>💡 INSIGHT:</strong> {insight}<br><br>
        <strong>📌 BUSINESS MEANING:</strong> {meaning}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================================
def render_executive_overview(df, kpis, insights):
    """Render the Executive Overview dashboard page."""
    st.markdown("## 📊 Executive Overview")
    st.markdown("*Management-level snapshot of business performance*")
    st.markdown("---")

    # --- KPI Cards Row 1 ---
    cols = st.columns(5)
    with cols[0]:
        render_kpi_card("Total Revenue",
                        f"${kpis.get('total_revenue', 0):,.2f}", "kpi-card")
    with cols[1]:
        render_kpi_card("Total Orders",
                        f"{kpis.get('total_orders', 0):,}", "kpi-card-green")
    with cols[2]:
        render_kpi_card("Total Customers",
                        f"{kpis.get('total_customers', 0):,}", "kpi-card-blue")
    with cols[3]:
        render_kpi_card("Avg Order Value",
                        f"${kpis.get('avg_order_value', 0):.2f}", "kpi-card-orange")
    with cols[4]:
        render_kpi_card("Avg Rating",
                        f"{kpis.get('avg_rating', 0):.2f} ⭐", "kpi-card")

    # --- KPI Cards Row 2 ---
    cols2 = st.columns(5)
    with cols2[0]:
        render_kpi_card("Repeat Rate",
                        f"{kpis.get('repeat_rate', 0):.1f}%", "kpi-card-green")
    with cols2[1]:
        render_kpi_card("Restaurants",
                        f"{kpis.get('total_restaurants', 0):,}", "kpi-card-blue")
    with cols2[2]:
        render_kpi_card("Avg Delivery Time",
                        f"{kpis.get('avg_delivery_time', 0):.0f} min", "kpi-card-orange")
    with cols2[3]:
        render_kpi_card("Weekend Orders",
                        f"{kpis.get('weekend_pct', 0):.1f}%", "kpi-card")
    with cols2[4]:
        render_kpi_card("Cuisines",
                        f"{kpis.get('total_cuisines', 0)}", "kpi-card-green")

    st.markdown("---")

    # --- Charts Row ---
    col1, col2 = st.columns(2)

    with col1:
        # Cuisine Revenue Contribution
        if 'cuisine_type' in df.columns and 'cost_of_the_order' in df.columns:
            cuisine_rev = df.groupby('cuisine_type')['cost_of_the_order'].sum().reset_index()
            cuisine_rev.columns = ['Cuisine', 'Revenue']
            cuisine_rev = cuisine_rev.sort_values('Revenue', ascending=False)

            fig = px.pie(cuisine_rev, values='Revenue', names='Cuisine',
                         title='Revenue by Cuisine Type',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
            fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Orders by Day of Week
        if 'day_of_the_week' in df.columns:
            day_orders = df['day_of_the_week'].value_counts().reset_index()
            day_orders.columns = ['Day', 'Orders']

            fig = px.bar(day_orders, x='Day', y='Orders',
                         title='Order Volume: Weekend vs Weekday',
                         color='Day',
                         color_discrete_map={'Weekend': '#667eea', 'Weekday': '#764ba2'})
            fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- Second Charts Row ---
    col3, col4 = st.columns(2)

    with col3:
        # Order Value Distribution
        if 'cost_of_the_order' in df.columns:
            fig = px.histogram(df, x='cost_of_the_order', nbins=30,
                               title='Order Value Distribution',
                               color_discrete_sequence=['#667eea'],
                               labels={'cost_of_the_order': 'Order Value ($)'})
            fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Rating Distribution
        if 'rating_numeric' in df.columns:
            rated = df[df['rating_given'] == True]
            rating_dist = rated['rating_numeric'].value_counts().sort_index().reset_index()
            rating_dist.columns = ['Rating', 'Count']

            # Add "Not Given"
            not_given = len(df) - len(rated)
            not_given_df = pd.DataFrame({'Rating': ['Not Given'], 'Count': [not_given]})
            rating_dist['Rating'] = rating_dist['Rating'].astype(str)
            rating_dist = pd.concat([rating_dist, not_given_df], ignore_index=True)

            fig = px.bar(rating_dist, x='Rating', y='Count',
                         title='Rating Distribution',
                         color='Rating',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- What Is Happening Section ---
    st.markdown("---")
    st.markdown("### 🔍 What Is Happening?")
    st.markdown("*Key findings from the data — auto-generated from actual metrics*")

    for insight in insights[:5]:
        card_type = {
            'strength': 'insight-card-success',
            'risk': 'insight-card-danger',
            'opportunity': 'insight-card-warning',
            'neutral': 'insight-card'
        }.get(insight.get('category', 'neutral'), 'insight-card')

        render_insight_card(
            insight['icon'],
            insight['fact'],
            insight['insight'],
            insight['meaning'],
            card_type
        )


# ============================================================================
# PAGE 2: SALES & RESTAURANT ANALYSIS
# ============================================================================
def render_sales_analysis(df, kpis):
    """Render the Sales & Restaurant Analysis page."""
    st.markdown("## 🏪 Sales & Restaurant Analysis")
    st.markdown("*Business performance by restaurant, cuisine, and patterns*")
    st.markdown("---")

    # --- Top Restaurants by Revenue ---
    col1, col2 = st.columns(2)

    with col1:
        if 'restaurant_name' in df.columns and 'cost_of_the_order' in df.columns:
            rest_rev = df.groupby('restaurant_name').agg(
                revenue=('cost_of_the_order', 'sum'),
                orders=('order_id', 'count'),
                avg_order=('cost_of_the_order', 'mean')
            ).sort_values('revenue', ascending=False).head(15).reset_index()

            fig = px.bar(rest_rev, x='revenue', y='restaurant_name',
                         title='Top 15 Restaurants by Revenue',
                         orientation='h',
                         color='revenue',
                         color_continuous_scale='Viridis',
                         labels={'revenue': 'Revenue ($)',
                                 'restaurant_name': 'Restaurant'})
            fig.update_layout(height=500, margin=dict(t=50, b=20, l=20, r=20),
                              yaxis={'categoryorder': 'total ascending'},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'restaurant_name' in df.columns:
            rest_orders = df.groupby('restaurant_name').agg(
                orders=('order_id', 'count'),
                revenue=('cost_of_the_order', 'sum'),
            ).sort_values('orders', ascending=False).head(15).reset_index()

            fig = px.bar(rest_orders, x='orders', y='restaurant_name',
                         title='Top 15 Restaurants by Order Volume',
                         orientation='h',
                         color='orders',
                         color_continuous_scale='Turbo',
                         labels={'orders': 'Orders',
                                 'restaurant_name': 'Restaurant'})
            fig.update_layout(height=500, margin=dict(t=50, b=20, l=20, r=20),
                              yaxis={'categoryorder': 'total ascending'},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- Cuisine Analysis ---
    st.markdown("### 🍽️ Cuisine Performance")
    col3, col4 = st.columns(2)

    with col3:
        if 'cuisine_type' in df.columns:
            cuisine_stats = df.groupby('cuisine_type').agg(
                orders=('order_id', 'count'),
                revenue=('cost_of_the_order', 'sum'),
                avg_order=('cost_of_the_order', 'mean'),
                avg_rating=('rating_numeric', 'mean')
            ).sort_values('revenue', ascending=False).reset_index()

            fig = px.bar(cuisine_stats, x='cuisine_type', y=['revenue'],
                         title='Revenue by Cuisine',
                         color='cuisine_type',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         labels={'cuisine_type': 'Cuisine', 'value': 'Revenue ($)'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Cuisine metrics table
            display_df = cuisine_stats.copy()
            display_df.columns = ['Cuisine', 'Orders', 'Revenue', 'Avg Order ($)', 'Avg Rating']
            display_df['Revenue'] = display_df['Revenue'].round(2)
            display_df['Avg Order ($)'] = display_df['Avg Order ($)'].round(2)
            display_df['Avg Rating'] = display_df['Avg Rating'].round(2)
            display_df['Revenue Share (%)'] = (
                display_df['Revenue'] / display_df['Revenue'].sum() * 100
            ).round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col4:
        # Cuisine by Average Order Value
        if 'cuisine_type' in df.columns:
            fig = px.bar(cuisine_stats, x='cuisine_type', y='avg_order',
                         title='Average Order Value by Cuisine',
                         color='avg_order',
                         color_continuous_scale='RdYlGn',
                         labels={'cuisine_type': 'Cuisine',
                                 'avg_order': 'Avg Order Value ($)'})
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Cuisine by Rating
        if 'rating_numeric' in df.columns:
            fig = px.bar(cuisine_stats, x='cuisine_type', y='avg_rating',
                         title='Average Rating by Cuisine',
                         color='avg_rating',
                         color_continuous_scale='RdYlGn',
                         labels={'cuisine_type': 'Cuisine',
                                 'avg_rating': 'Avg Rating'})
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- Pareto Analysis ---
    st.markdown("### 📊 Pareto Analysis (80/20 Rule)")

    if 'restaurant_name' in df.columns and 'cost_of_the_order' in df.columns:
        rest_rev = df.groupby('restaurant_name')['cost_of_the_order'].sum().sort_values(
            ascending=False
        ).reset_index()
        rest_rev.columns = ['Restaurant', 'Revenue']
        rest_rev['Cumulative Revenue'] = rest_rev['Revenue'].cumsum()
        rest_rev['Cumulative %'] = (
            rest_rev['Cumulative Revenue'] / rest_rev['Revenue'].sum() * 100
        )
        rest_rev['Restaurant %'] = (
            np.arange(1, len(rest_rev) + 1) / len(rest_rev) * 100
        )

        # Find the 80% revenue point
        idx_80 = rest_rev[rest_rev['Cumulative %'] >= 80].index[0] + 1
        pct_restaurants_80 = idx_80 / len(rest_rev) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=rest_rev['Restaurant'][:30],
                   y=rest_rev['Revenue'][:30],
                   name='Revenue',
                   marker_color='#667eea'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=rest_rev['Restaurant'][:30],
                       y=rest_rev['Cumulative %'][:30],
                       name='Cumulative %',
                       line=dict(color='#f5576c', width=2),
                       mode='lines+markers'),
            secondary_y=True
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% Revenue",
                      secondary_y=True)
        fig.update_layout(
            title=f'Pareto Analysis: {pct_restaurants_80:.1f}% of restaurants '
                  f'generate 80% of revenue',
            height=450,
            xaxis_tickangle=-45
        )
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-card">
        <strong>📊 Pareto Insight:</strong> {pct_restaurants_80:.1f}% of restaurants
        ({idx_80} out of {len(rest_rev)}) generate 80% of total revenue.
        This confirms the classic 80/20 pattern and highlights the importance of
        maintaining strong relationships with top-performing restaurant partners.
        </div>
        """, unsafe_allow_html=True)

    # --- Revenue by Order Value Category ---
    st.markdown("### 💰 Order Value Segmentation")
    if 'order_value_category' in df.columns:
        col5, col6 = st.columns(2)
        with col5:
            ov_stats = df.groupby('order_value_category', observed=True).agg(
                orders=('order_id', 'count'),
                revenue=('cost_of_the_order', 'sum')
            ).reset_index()

            fig = px.pie(ov_stats, values='orders', names='order_value_category',
                         title='Order Distribution by Value Category',
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         hole=0.3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            fig = px.pie(ov_stats, values='revenue', names='order_value_category',
                         title='Revenue Distribution by Value Category',
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         hole=0.3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- Business Driver Summary ---
    st.markdown("### 🎯 Key Business Drivers")
    if 'cuisine_type' in df.columns and 'cost_of_the_order' in df.columns:
        top_cuisine_name = df.groupby('cuisine_type')['cost_of_the_order'].sum().idxmax()
        top_cuisine_rev = df.groupby('cuisine_type')['cost_of_the_order'].sum().max()
        top_rest_name = df.groupby('restaurant_name')['cost_of_the_order'].sum().idxmax()
        top_rest_rev = df.groupby('restaurant_name')['cost_of_the_order'].sum().max()

        col7, col8, col9 = st.columns(3)
        with col7:
            st.metric("🍽️ Top Cuisine", top_cuisine_name,
                      f"${top_cuisine_rev:,.2f}")
        with col8:
            st.metric("🏪 Top Restaurant", top_rest_name,
                      f"${top_rest_rev:,.2f}")
        with col9:
            st.metric("💵 Median Order", f"${kpis.get('median_order_value', 0):.2f}",
                      f"Mean: ${kpis.get('avg_order_value', 0):.2f}")


# ============================================================================
# PAGE 3: CUSTOMER INTELLIGENCE
# ============================================================================
def render_customer_intelligence(df, customer_df, kpis):
    """Render the Customer Intelligence page."""
    st.markdown("## 👥 Customer Intelligence")
    st.markdown("*Deep dive into customer behavior, segmentation, and value*")
    st.markdown("---")

    if customer_df is None:
        st.warning("Customer analysis not available — customer_id column missing.")
        return

    # --- Customer KPI Summary ---
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card("Total Customers",
                        f"{len(customer_df):,}", "kpi-card-blue")
    with cols[1]:
        repeat = (customer_df['total_orders'] > 1).sum()
        render_kpi_card("Repeat Customers",
                        f"{repeat:,} ({repeat/len(customer_df)*100:.1f}%)",
                        "kpi-card-green")
    with cols[2]:
        render_kpi_card("Avg Orders/Customer",
                        f"{customer_df['total_orders'].mean():.2f}",
                        "kpi-card-orange")
    with cols[3]:
        render_kpi_card("Avg Revenue/Customer",
                        f"${customer_df['total_spend'].mean():.2f}",
                        "kpi-card")

    st.markdown("---")

    # --- Customer Segmentation ---
    st.markdown("### 🏆 Customer Segments (FME Analysis)")
    st.caption("*Frequency-Monetary-Engagement segmentation adapted for available data*")

    col1, col2 = st.columns(2)

    with col1:
        if 'segment' in customer_df.columns:
            seg_counts = customer_df['segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Customers']

            colors_map = {
                'Champions': '#4caf50',
                'Loyal Customers': '#8bc34a',
                'Potential Loyalists': '#03a9f4',
                'At Risk': '#ff9800',
                'Lost / Inactive': '#f44336',
                'Need Attention': '#ff5722',
                'New / Occasional': '#9e9e9e'
            }

            fig = px.pie(seg_counts, values='Customers', names='Segment',
                         title='Customer Segment Distribution',
                         color='Segment',
                         color_discrete_map=colors_map,
                         hole=0.4)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'segment' in customer_df.columns:
            seg_rev = customer_df.groupby('segment')['total_spend'].sum().reset_index()
            seg_rev.columns = ['Segment', 'Revenue']
            seg_rev = seg_rev.sort_values('Revenue', ascending=True)

            fig = px.bar(seg_rev, x='Revenue', y='Segment',
                         title='Revenue Contribution by Segment',
                         orientation='h',
                         color='Segment',
                         color_discrete_map=colors_map)
            fig.update_layout(height=450, showlegend=False,
                              xaxis_title='Total Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)

    # --- Segment Detail Table ---
    if 'segment' in customer_df.columns:
        st.markdown("### 📋 Segment Summary")
        seg_summary = customer_df.groupby('segment').agg(
            customers=('customer_id', 'count'),
            total_revenue=('total_spend', 'sum'),
            avg_revenue=('total_spend', 'mean'),
            avg_orders=('total_orders', 'mean'),
            avg_delivery_time=('avg_delivery_time', 'mean'),
        ).reset_index()

        seg_summary['revenue_share_pct'] = (
            seg_summary['total_revenue'] / seg_summary['total_revenue'].sum() * 100
        ).round(1)
        seg_summary = seg_summary.sort_values('total_revenue', ascending=False)

        display_seg = seg_summary.copy()
        display_seg.columns = ['Segment', 'Customers', 'Total Revenue',
                               'Avg Revenue', 'Avg Orders',
                               'Avg Delivery (min)', 'Revenue Share (%)']
        for col in ['Total Revenue', 'Avg Revenue']:
            display_seg[col] = display_seg[col].apply(lambda x: f"${x:,.2f}")
        display_seg['Avg Orders'] = display_seg['Avg Orders'].round(2)
        display_seg['Avg Delivery (min)'] = display_seg['Avg Delivery (min)'].round(1)

        st.dataframe(display_seg, use_container_width=True, hide_index=True)

    # --- Orders per Customer Distribution ---
    st.markdown("### 📊 Customer Behavior Patterns")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(customer_df, x='total_orders',
                           title='Orders per Customer Distribution',
                           color_discrete_sequence=['#667eea'],
                           labels={'total_orders': 'Number of Orders'},
                           nbins=int(max(customer_df['total_orders'].max(), 10)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(customer_df, x='total_spend',
                           title='Revenue per Customer Distribution',
                           color_discrete_sequence=['#764ba2'],
                           labels={'total_spend': 'Total Spend ($)'},
                           nbins=30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Customer Value Tiers ---
    st.markdown("### 💎 Customer Value Tiers")
    col5, col6 = st.columns(2)

    with col5:
        if 'value_tier' in customer_df.columns:
            tier_stats = customer_df.groupby('value_tier', observed=True).agg(
                customers=('customer_id', 'count'),
                revenue=('total_spend', 'sum')
            ).reset_index()

            fig = px.bar(tier_stats, x='value_tier', y='customers',
                         title='Customers by Value Tier',
                         color='value_tier',
                         color_discrete_sequence=['#f44336', '#ff9800',
                                                  '#4caf50', '#667eea'])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        if 'value_tier' in customer_df.columns:
            fig = px.bar(tier_stats, x='value_tier', y='revenue',
                         title='Revenue by Value Tier',
                         color='value_tier',
                         color_discrete_sequence=['#f44336', '#ff9800',
                                                  '#4caf50', '#667eea'])
            fig.update_layout(height=400, showlegend=False,
                              yaxis_title='Total Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)

    # --- Cuisine Diversity vs Spend ---
    st.markdown("### 🍽️ Cuisine Exploration vs Spending")
    fig = px.scatter(customer_df, x='cuisines_tried', y='total_spend',
                     size='total_orders', color='segment' if 'segment' in customer_df.columns else None,
                     title='Customer Spend vs Cuisine Diversity',
                     labels={'cuisines_tried': 'Cuisines Tried',
                             'total_spend': 'Total Spend ($)',
                             'total_orders': 'Total Orders'},
                     hover_data=['customer_id', 'total_orders'],
                     color_discrete_map=colors_map if 'segment' in customer_df.columns else None)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 Customers Table ---
    st.markdown("### 🥇 Top 10 Customers by Revenue")
    top_customers = customer_df.nlargest(10, 'total_spend')[
        ['customer_id', 'total_orders', 'total_spend', 'avg_order_value',
         'cuisines_tried', 'restaurants_tried', 'segment']
    ].copy()
    top_customers['total_spend'] = top_customers['total_spend'].apply(
        lambda x: f"${x:,.2f}"
    )
    top_customers['avg_order_value'] = top_customers['avg_order_value'].apply(
        lambda x: f"${x:,.2f}"
    )
    st.dataframe(top_customers, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 4: DELIVERY & OPERATIONS
# ============================================================================
def render_delivery_operations(df, kpis):
    """Render the Delivery & Operations analysis page."""
    st.markdown("## 🚚 Delivery & Operations Analysis")
    st.markdown("*Operational performance and bottleneck identification*")
    st.markdown("---")

    # --- Operational KPIs ---
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card("Avg Prep Time",
                        f"{kpis.get('avg_prep_time', 0):.0f} min",
                        "kpi-card-orange")
    with cols[1]:
        render_kpi_card("Avg Delivery Time",
                        f"{kpis.get('avg_delivery_time', 0):.0f} min",
                        "kpi-card-blue")
    with cols[2]:
        render_kpi_card("Avg Total Time",
                        f"{kpis.get('avg_total_time', 0):.0f} min",
                        "kpi-card")
    with cols[3]:
        rating_pct = kpis.get('rating_given_pct', 0)
        render_kpi_card("Rating Response",
                        f"{rating_pct:.1f}%",
                        "kpi-card-green")

    st.markdown("---")

    # --- Time Distribution Analysis ---
    st.markdown("### ⏱️ Time Distribution Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if 'food_preparation_time' in df.columns:
            fig = px.histogram(df, x='food_preparation_time', nbins=25,
                               title='Food Preparation Time Distribution',
                               color_discrete_sequence=['#ff9800'],
                               labels={'food_preparation_time': 'Prep Time (min)'})
            fig.add_vline(x=kpis.get('avg_prep_time', 0),
                          line_dash="dash", line_color="red",
                          annotation_text=f"Avg: {kpis.get('avg_prep_time', 0):.0f}min")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'delivery_time' in df.columns:
            fig = px.histogram(df, x='delivery_time', nbins=25,
                               title='Delivery Time Distribution',
                               color_discrete_sequence=['#4facfe'],
                               labels={'delivery_time': 'Delivery Time (min)'})
            fig.add_vline(x=kpis.get('avg_delivery_time', 0),
                          line_dash="dash", line_color="red",
                          annotation_text=f"Avg: {kpis.get('avg_delivery_time', 0):.0f}min")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- Cuisine-wise Operations ---
    st.markdown("### 🍽️ Operational Performance by Cuisine")
    if 'cuisine_type' in df.columns:
        ops_by_cuisine = df.groupby('cuisine_type').agg(
            avg_prep=('food_preparation_time', 'mean'),
            avg_delivery=('delivery_time', 'mean'),
            avg_total=('total_time', 'mean'),
            avg_rating=('rating_numeric', 'mean'),
            orders=('order_id', 'count')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Prep Time', x=ops_by_cuisine['cuisine_type'],
            y=ops_by_cuisine['avg_prep'], marker_color='#ff9800'
        ))
        fig.add_trace(go.Bar(
            name='Delivery Time', x=ops_by_cuisine['cuisine_type'],
            y=ops_by_cuisine['avg_delivery'], marker_color='#4facfe'
        ))
        fig.update_layout(
            title='Average Prep & Delivery Time by Cuisine',
            barmode='stack', height=400,
            yaxis_title='Time (minutes)',
            xaxis_title='Cuisine'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Prep Time vs Delivery Time Scatter ---
    st.markdown("### 🔄 Preparation vs Delivery Time Relationship")
    col3, col4 = st.columns(2)

    with col3:
        if 'food_preparation_time' in df.columns and 'delivery_time' in df.columns:
            fig = px.scatter(df, x='food_preparation_time', y='delivery_time',
                             color='cuisine_type' if 'cuisine_type' in df.columns else None,
                             title='Prep Time vs Delivery Time',
                             labels={'food_preparation_time': 'Prep Time (min)',
                                     'delivery_time': 'Delivery Time (min)'},
                             opacity=0.5)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Rating vs Total Time
        if 'total_time' in df.columns and 'rating_numeric' in df.columns:
            rated_df = df.dropna(subset=['rating_numeric'])
            if len(rated_df) > 0:
                fig = px.box(rated_df, x='rating_numeric', y='total_time',
                             title='Total Time by Customer Rating',
                             labels={'rating_numeric': 'Rating',
                                     'total_time': 'Total Time (min)'},
                             color='rating_numeric',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # --- Weekend vs Weekday Operations ---
    st.markdown("### 📅 Weekend vs Weekday Operations")
    if 'day_of_the_week' in df.columns:
        day_ops = df.groupby('day_of_the_week').agg(
            avg_prep=('food_preparation_time', 'mean'),
            avg_delivery=('delivery_time', 'mean'),
            avg_total=('total_time', 'mean'),
            avg_cost=('cost_of_the_order', 'mean'),
            avg_rating=('rating_numeric', 'mean'),
            orders=('order_id', 'count')
        ).reset_index()

        cols = st.columns(3)
        for i, (day, data) in enumerate(day_ops.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                **{data['day_of_the_week']}** ({data['orders']:,} orders)
                - Avg Prep: {data['avg_prep']:.1f} min
                - Avg Delivery: {data['avg_delivery']:.1f} min
                - Avg Total: {data['avg_total']:.1f} min
                - Avg Order Value: ${data['avg_cost']:.2f}
                - Avg Rating: {data['avg_rating']:.2f}
                """)

    # --- Delivery Speed Analysis ---
    st.markdown("### 🏎️ Delivery Speed Categories")
    if 'delivery_speed' in df.columns:
        col5, col6 = st.columns(2)
        with col5:
            speed_stats = df.groupby('delivery_speed', observed=True).agg(
                orders=('order_id', 'count'),
                avg_rating=('rating_numeric', 'mean')
            ).reset_index()

            fig = px.bar(speed_stats, x='delivery_speed', y='orders',
                         title='Orders by Delivery Speed',
                         color='delivery_speed',
                         color_discrete_sequence=['#4caf50', '#8bc34a',
                                                  '#ff9800', '#f44336'])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            fig = px.bar(speed_stats, x='delivery_speed', y='avg_rating',
                         title='Avg Rating by Delivery Speed',
                         color='delivery_speed',
                         color_discrete_sequence=['#4caf50', '#8bc34a',
                                                  '#ff9800', '#f44336'])
            fig.update_layout(height=400, showlegend=False,
                              yaxis_title='Average Rating')
            st.plotly_chart(fig, use_container_width=True)

    # --- Operational Bottleneck Summary ---
    st.markdown("### ⚠️ Operational Bottleneck Summary")

    if 'restaurant_name' in df.columns:
        rest_ops = df.groupby('restaurant_name').agg(
            orders=('order_id', 'count'),
            avg_prep=('food_preparation_time', 'mean'),
            avg_delivery=('delivery_time', 'mean'),
            avg_total=('total_time', 'mean'),
            avg_rating=('rating_numeric', 'mean')
        ).reset_index()

        # Slowest restaurants (minimum 5 orders)
        slow_rest = rest_ops[rest_ops['orders'] >= 5].nlargest(10, 'avg_total')
        if len(slow_rest) > 0:
            st.markdown("**🐌 Slowest Restaurants (min 5 orders):**")
            display_slow = slow_rest[['restaurant_name', 'orders', 'avg_prep',
                                      'avg_delivery', 'avg_total', 'avg_rating']].copy()
            display_slow.columns = ['Restaurant', 'Orders', 'Avg Prep (min)',
                                    'Avg Delivery (min)', 'Avg Total (min)', 'Avg Rating']
            for col in ['Avg Prep (min)', 'Avg Delivery (min)', 'Avg Total (min)', 'Avg Rating']:
                display_slow[col] = display_slow[col].round(1)
            st.dataframe(display_slow, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 5: AI RISK ANALYSIS
# ============================================================================
def render_ai_risk_analysis(model_output, customer_df_with_risk):
    """Render the AI/ML Customer Risk Analysis page."""
    st.markdown("## 🤖 AI-Assisted Customer Risk Analysis")
    st.markdown("*Machine learning-powered customer risk scoring and prediction*")
    st.markdown("---")

    if model_output is None:
        st.warning("⚠️ Insufficient data for AI risk analysis. "
                   "The dataset needs more customer history for meaningful predictions.")
        return

    # --- Model Information ---
    st.markdown("### 📋 Model Overview")
    st.markdown(f"""
    <div class="insight-card">
    <strong>Model Type:</strong> {model_output['model_name']}<br>
    <strong>Approach:</strong> AI-Assisted Customer Risk Scoring — identifies customers
    showing behavioral patterns associated with disengagement (low frequency, low engagement,
    poor service experience).<br>
    <strong>Training Data:</strong> {model_output['train_size']} customers (75%) |
    Testing Data: {model_output['test_size']} customers (25%)<br>
    <strong>Features Used:</strong> {', '.join(model_output['feature_cols'])}
    </div>
    """, unsafe_allow_html=True)

    st.caption("⚠️ *Note: This is a behavioral risk scoring system based on available "
               "order data. It is NOT a validated churn prediction model since the dataset "
               "does not contain temporal/date information for true churn definition.*")

    # --- Model Performance Metrics ---
    st.markdown("### 📊 Model Performance")
    metrics = model_output['best_metrics']

    cols = st.columns(5)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
        with cols[i]:
            val = metrics.get(key, 0)
            color = "kpi-card-green" if val >= 0.7 else (
                "kpi-card-orange" if val >= 0.5 else "kpi-card"
            )
            render_kpi_card(name, f"{val:.3f}", color)

    # --- All Models Comparison ---
    st.markdown("### 🔬 Model Comparison")
    model_comp = []
    for name, result in model_output['results'].items():
        model_comp.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'Recall': f"{result['recall']:.3f}",
            'F1-Score': f"{result['f1']:.3f}",
            'ROC-AUC': f"{result['roc_auc']:.3f}"
        })
    st.dataframe(pd.DataFrame(model_comp), use_container_width=True, hide_index=True)

    st.markdown(f"**Selected Model:** {model_output['model_name']} "
                f"(best F1-Score: {metrics['f1']:.3f})")

    # --- Feature Importance ---
    st.markdown("### 🎯 Feature Importance")
    col1, col2 = st.columns([2, 1])

    with col1:
        fi = model_output['feature_importance']
        fig = px.bar(fi, x='importance', y='feature',
                     title='What Drives Customer Risk?',
                     orientation='h',
                     color='importance',
                     color_continuous_scale='RdYlGn_r',
                     labels={'importance': 'Importance Score',
                             'feature': 'Feature'})
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'},
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Feature Explanation:**")
        feature_desc = {
            'total_orders': 'Number of orders placed by customer',
            'total_spend': 'Total amount spent ($)',
            'avg_order_value': 'Average value per order ($)',
            'avg_delivery_time': 'Average delivery wait time (min)',
            'avg_prep_time': 'Average food prep time (min)',
            'cuisines_tried': 'Number of different cuisines ordered',
            'restaurants_tried': 'Number of different restaurants used',
            'rating_response_rate': 'Percentage of orders with ratings',
            'avg_rating': 'Average rating given',
            'weekend_order_pct': 'Percentage of orders on weekends'
        }
        for _, row in fi.iterrows():
            desc = feature_desc.get(row['feature'], row['feature'])
            st.markdown(f"- **{row['feature']}**: {desc}")

    # --- Risk Distribution ---
    st.markdown("### 📈 Customer Risk Distribution")
    col3, col4 = st.columns(2)

    with col3:
        if customer_df_with_risk is not None and 'risk_category' in customer_df_with_risk.columns:
            risk_dist = customer_df_with_risk['risk_category'].value_counts().reset_index()
            risk_dist.columns = ['Risk Level', 'Customers']

            risk_colors = {'High Risk': '#f44336', 'Medium Risk': '#ff9800',
                           'Low Risk': '#4caf50'}

            fig = px.pie(risk_dist, values='Customers', names='Risk Level',
                         title='Customer Risk Distribution',
                         color='Risk Level',
                         color_discrete_map=risk_colors,
                         hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if customer_df_with_risk is not None and 'risk_probability' in customer_df_with_risk.columns:
            fig = px.histogram(customer_df_with_risk, x='risk_probability',
                               nbins=30,
                               title='Risk Probability Distribution',
                               color_discrete_sequence=['#667eea'],
                               labels={'risk_probability': 'Risk Probability'})
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                          annotation_text="Risk Threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- High Risk Customers Table ---
    st.markdown("### ⚠️ High-Risk Customers")
    if customer_df_with_risk is not None:
        high_risk = customer_df_with_risk[
            customer_df_with_risk['risk_category'] == 'High Risk'
        ].nlargest(20, 'risk_probability')

        if len(high_risk) > 0:
            display_risk = high_risk[['customer_id', 'risk_probability',
                                      'total_orders', 'total_spend',
                                      'segment']].copy()
            display_risk['risk_probability'] = display_risk['risk_probability'].apply(
                lambda x: f"{x:.1%}"
            )
            display_risk['total_spend'] = display_risk['total_spend'].apply(
                lambda x: f"${x:,.2f}"
            )
            display_risk.columns = ['Customer ID', 'Risk Probability',
                                    'Total Orders', 'Total Spend', 'Segment']
            st.dataframe(display_risk, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="insight-card-warning">
            <strong>⚠️ Action Required:</strong> {len(high_risk)} customers identified as
            High Risk. These customers show behavioral patterns associated with disengagement:
            low order frequency, limited cuisine diversity, and/or poor service experiences.
            <strong>Recommended:</strong> Prioritize these customers for targeted retention campaigns.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No high-risk customers identified in the current dataset.")

    # --- Risk by Segment Cross-Analysis ---
    st.markdown("### 🔄 Risk vs Customer Segment")
    if customer_df_with_risk is not None and 'segment' in customer_df_with_risk.columns:
        risk_seg = customer_df_with_risk.groupby(
            ['segment', 'risk_category'], observed=True
        ).size().reset_index(name='count')

        fig = px.bar(risk_seg, x='segment', y='count', color='risk_category',
                     title='Risk Distribution Across Customer Segments',
                     color_discrete_map={'High Risk': '#f44336',
                                         'Medium Risk': '#ff9800',
                                         'Low Risk': '#4caf50'},
                     barmode='stack')
        fig.update_layout(height=400, xaxis_title='Customer Segment',
                          yaxis_title='Number of Customers')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 6: OPPORTUNITIES & ACTIONS
# ============================================================================
def render_opportunities_actions(recommendations, kpis, model_output=None):
    """Render the Opportunities & Actions page."""
    st.markdown("## 🎯 Opportunities & Recommended Actions")
    st.markdown("*Converting data insights into business decisions*")
    st.markdown("---")

    if not recommendations:
        st.info("No recommendations generated — insufficient analysis data.")
        return

    # --- Impact vs Urgency Matrix ---
    st.markdown("### 📊 Impact vs Urgency Matrix")

    matrix_data = []
    for rec in recommendations:
        impact_val = {'High': 3, 'Medium': 2, 'Low': 1}.get(rec['impact'], 2)
        urgency_val = {'High': 3, 'Medium': 2, 'Low': 1}.get(rec['urgency'], 2)
        matrix_data.append({
            'Action': rec['title'],
            'Impact': impact_val,
            'Urgency': urgency_val,
            'Impact_Label': rec['impact'],
            'Urgency_Label': rec['urgency'],
            'Category': rec['category']
        })

    matrix_df = pd.DataFrame(matrix_data)

    fig = px.scatter(matrix_df, x='Urgency', y='Impact',
                     text='Action', size=[30]*len(matrix_df),
                     color='Category',
                     title='Action Priority Matrix',
                     labels={'Urgency': 'Urgency →', 'Impact': 'Impact →'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_xaxes(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'],
                     range=[0.5, 3.5])
    fig.update_yaxes(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'],
                     range=[0.5, 3.5])

    # Add quadrant labels
    fig.add_annotation(x=2.8, y=2.8, text="🔴 DO NOW",
                       showarrow=False, font=dict(size=14, color='red'))
    fig.add_annotation(x=1.2, y=2.8, text="🟢 PLAN",
                       showarrow=False, font=dict(size=14, color='green'))
    fig.add_annotation(x=2.8, y=1.2, text="🟡 DELEGATE",
                       showarrow=False, font=dict(size=14, color='orange'))
    fig.add_annotation(x=1.2, y=1.2, text="⚪ CONSIDER",
                       showarrow=False, font=dict(size=14, color='gray'))

    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Priority Groups ---
    st.markdown("### 🚀 Prioritized Action Plan")

    # Group by priority
    priority_order = [
        ('🔴 High Impact / High Urgency — DO NOW', 'High', 'High'),
        ('🟢 High Impact / Low Urgency — PLAN', 'High', 'Medium'),
        ('🟢 High Impact / Low Urgency — PLAN', 'High', 'Low'),
        ('🟡 Medium Impact — DELEGATE', 'Medium', 'High'),
        ('🟡 Medium Impact — DELEGATE', 'Medium', 'Medium'),
        ('⚪ Lower Priority — CONSIDER', 'Medium', 'Low'),
        ('⚪ Lower Priority — CONSIDER', 'Low', 'Low'),
    ]

    displayed_groups = set()
    for group_name, impact, urgency in priority_order:
        matching = [r for r in recommendations
                    if r['impact'] == impact and r['urgency'] == urgency]

        if matching and group_name not in displayed_groups:
            st.markdown(f"#### {group_name}")
            displayed_groups.add(group_name)

            for rec in matching:
                with st.expander(f"📌 {rec['title']} [{rec['category']}]", expanded=True):
                    st.markdown(f"""
                    **📋 FACT:** {rec['fact']}

                    **💡 INSIGHT:** {rec['insight']}

                    **⚡ RISK/OPPORTUNITY:** {rec['risk_opportunity']}

                    **✅ RECOMMENDED ACTION:** {rec['action']}

                    ---
                    *Impact: {rec['impact']} | Urgency: {rec['urgency']} | Category: {rec['category']}*
                    """)

    # --- Summary Statistics ---
    st.markdown("### 📈 Recommendation Summary")
    cols = st.columns(3)

    high_priority = len([r for r in recommendations
                         if r['impact'] == 'High' and r['urgency'] == 'High'])
    with cols[0]:
        render_kpi_card("Total Recommendations",
                        f"{len(recommendations)}", "kpi-card-blue")
    with cols[1]:
        render_kpi_card("High Priority",
                        f"{high_priority}", "kpi-card-orange")
    with cols[2]:
        categories = set(r['category'] for r in recommendations)
        render_kpi_card("Categories Covered",
                        f"{len(categories)}", "kpi-card-green")


# ============================================================================
# PAGE 7: DATA QUALITY & METHODOLOGY
# ============================================================================
def render_data_quality(df, cleaning_report, data_info, kpis, model_output=None):
    """Render the Data Quality & Methodology page."""
    st.markdown("## 📋 Data Quality & Methodology")
    st.markdown("*Transparency in data processing and analysis methods*")
    st.markdown("---")

    # --- Dataset Information ---
    st.markdown("### 📁 Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Dataset Name** | FoodHub Order Analysis |
        | **Source** | {data_info['url']} |
        | **Load Method** | {data_info['load_method']} |
        | **Original Rows** | {data_info['original_rows']:,} |
        | **Original Columns** | {data_info['original_cols']} |
        | **Cleaned Rows** | {cleaning_report['final_rows']:,} |
        """)

    with col2:
        st.markdown("**Column Details:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    # --- Data Cleaning Report ---
    st.markdown("### 🧹 Data Cleaning Summary")

    clean_metrics = {
        'Original Records': f"{cleaning_report['original_rows']:,}",
        'Final Records': f"{cleaning_report['final_rows']:,}",
        'Records Removed': f"{cleaning_report['original_rows'] - cleaning_report['final_rows']:,}",
        'Duplicates Found': f"{cleaning_report['duplicates_found']:,}",
        'Duplicates Removed': f"{cleaning_report['duplicates_removed']:,}",
        'Ratings "Not Given"': f"{cleaning_report['rating_not_given_count']:,}",
        'Invalid Cost Records': f"{cleaning_report['invalid_records']:,}",
        'Cost Outliers Flagged': f"{cleaning_report['outliers_handled']:,}"
    }

    clean_df = pd.DataFrame(list(clean_metrics.items()),
                            columns=['Metric', 'Value'])
    st.dataframe(clean_df, use_container_width=True, hide_index=True)

    # --- Missing Values ---
    st.markdown("### 🔍 Missing Values Analysis")
    if cleaning_report['missing_values']:
        missing = pd.DataFrame(list(cleaning_report['missing_values'].items()),
                               columns=['Column', 'Missing Count'])
        missing['Missing %'] = (missing['Missing Count'] /
                                 cleaning_report['original_rows'] * 100).round(2)
        st.dataframe(missing, use_container_width=True, hide_index=True)

        if missing['Missing Count'].sum() == 0:
            st.success("✅ No missing values in the original dataset.")
        else:
            st.warning(f"⚠️ {missing['Missing Count'].sum()} total missing values found.")

    # --- Feature Engineering ---
    st.markdown("### ⚙️ Feature Engineering")
    features_created = [
        ('rating_numeric', 'Numeric conversion of rating (NaN for "Not given")'),
        ('rating_given', 'Boolean flag for whether rating was provided'),
        ('total_time', 'food_preparation_time + delivery_time'),
        ('cost_outlier', 'IQR-based outlier flag for order cost'),
        ('order_value_category', 'Binned order value (Budget/Economy/Standard/Premium/Luxury)'),
        ('delivery_speed', 'Binned delivery time (Fast/Normal/Slow/Very Slow)'),
        ('prep_speed', 'Binned preparation time categories'),
        ('is_weekend', 'Binary weekend indicator'),
        ('rating_category', 'Categorized rating (High/Medium/Low/Not Rated)')
    ]

    customer_features = [
        ('total_orders', 'Count of orders per customer'),
        ('total_spend', 'Sum of all order values per customer'),
        ('avg_order_value', 'Mean order value per customer'),
        ('cuisines_tried', 'Count of unique cuisines ordered'),
        ('restaurants_tried', 'Count of unique restaurants used'),
        ('frequency_score', 'Quantile-based frequency score (1-5)'),
        ('monetary_score', 'Quantile-based monetary score (1-5)'),
        ('engagement_score', 'Composite engagement metric (1-5)'),
        ('segment', 'Customer segment label (Champions, Loyal, etc.)'),
        ('value_tier', 'Customer value tier (Low/Medium/High/Premium)')
    ]

    st.markdown("**Order-Level Features:**")
    feat_df = pd.DataFrame(features_created, columns=['Feature', 'Description'])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("**Customer-Level Features:**")
    cust_feat_df = pd.DataFrame(customer_features, columns=['Feature', 'Description'])
    st.dataframe(cust_feat_df, use_container_width=True, hide_index=True)

    # --- KPI Methodology ---
    st.markdown("### 📏 KPI Definitions & Methodology")
    kpi_defs = [
        ('Total Revenue', 'Sum of cost_of_the_order for all orders', 'SUM(cost_of_the_order)'),
        ('Average Order Value', 'Mean order value across all orders', 'MEAN(cost_of_the_order)'),
        ('Total Customers', 'Count of unique customer_id values', 'COUNT(DISTINCT customer_id)'),
        ('Repeat Rate', 'Percentage of customers with >1 order', '(Customers with orders > 1) / Total Customers × 100'),
        ('Average Rating', 'Mean of numeric ratings (excluding "Not given")', 'MEAN(rating WHERE rating ≠ "Not given")'),
        ('Rating Response Rate', 'Percentage of orders with a rating', '(Rated orders / Total orders) × 100'),
        ('Average Delivery Time', 'Mean delivery_time across all orders', 'MEAN(delivery_time)'),
        ('Average Prep Time', 'Mean food_preparation_time', 'MEAN(food_preparation_time)')
    ]

    kpi_df = pd.DataFrame(kpi_defs, columns=['KPI', 'Definition', 'Formula'])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    # --- ML Methodology ---
    st.markdown("### 🤖 AI/ML Methodology")
    if model_output:
        st.markdown(f"""
        **Approach:** AI-Assisted Customer Risk Scoring

        **Why Not Traditional Churn Prediction?**
        The FoodHub dataset does not contain date/timestamp information for orders.
        Without temporal data, we cannot define a true "churn window" (e.g., no orders
        in the last 30/60/90 days). Therefore, we implement a behavioral risk scoring
        system that identifies customers exhibiting patterns associated with disengagement.

        **Risk Label Definition:**
        A customer is labeled "at-risk" based on a composite of behavioral signals:
        - Low order frequency (bottom 30th percentile)
        - Low total spend (bottom 30th percentile)
        - Low rating response rate (<30%)
        - Higher-than-average delivery times
        - Limited cuisine diversity (only 1 cuisine tried)

        **Models Evaluated:**
        - Logistic Regression (with feature scaling)
        - Random Forest (100 estimators, max_depth=5)
        - Gradient Boosting (100 estimators, max_depth=3)

        **Best Model:** {model_output['model_name']}

        **Train/Test Split:** 75% / 25% (stratified by risk label)

        **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
        """)
    else:
        st.info("ML model was not trained — insufficient customer data.")

    # --- Limitations ---
    st.markdown("### ⚠️ Known Limitations")
    limitations = [
        "No date/timestamp data — cannot analyze temporal trends or true churn windows.",
        "No geographic/location data — cannot perform location-based analysis.",
        "No order status data — cannot analyze cancellations or failures.",
        "No payment method data — cannot analyze payment preferences.",
        "No discount/promotion data — cannot assess promotional impact.",
        "Ratings have ~36% 'Not given' — satisfaction metrics may be biased toward vocal customers.",
        "Dataset size (~1,898 orders) is relatively small for production ML models.",
        "Customer identity is based on customer_id only — no demographic information.",
        "Risk scoring is behavioral, not true churn prediction (no temporal data).",
        "No revenue target or budget data for ROI calculations."
    ]
    for i, lim in enumerate(limitations, 1):
        st.markdown(f"{i}. {lim}")

    # --- Data Sample ---
    st.markdown("### 📄 Data Sample (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ============================================================================
# SIDEBAR & NAVIGATION
# ============================================================================
def render_sidebar(df):
    """Render the sidebar with navigation and filters."""
    with st.sidebar:
        st.markdown("# 🍕 FoodHub BI")
        st.markdown("#### Business Intelligence Dashboard")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "📌 Navigation",
            [
                "📊 Executive Overview",
                "🏪 Sales & Restaurant Analysis",
                "👥 Customer Intelligence",
                "🚚 Delivery & Operations",
                "🤖 AI Risk Analysis",
                "🎯 Opportunities & Actions",
                "📋 Data Quality & Methodology"
            ],
            index=0
        )

        st.markdown("---")

        # --- Filters ---
        st.markdown("### 🔍 Filters")

        # Cuisine filter
        selected_cuisines = None
        if 'cuisine_type' in df.columns:
            cuisine_options = ['All'] + sorted(df['cuisine_type'].unique().tolist())
            selected_cuisines = st.multiselect(
                "Cuisine Type",
                options=df['cuisine_type'].unique().tolist(),
                default=df['cuisine_type'].unique().tolist(),
                key='cuisine_filter'
            )

        # Day filter
        selected_days = None
        if 'day_of_the_week' in df.columns:
            selected_days = st.multiselect(
                "Day of Week",
                options=df['day_of_the_week'].unique().tolist(),
                default=df['day_of_the_week'].unique().tolist(),
                key='day_filter'
            )

        # Cost range filter
        cost_range = None
        if 'cost_of_the_order' in df.columns:
            min_cost = float(df['cost_of_the_order'].min())
            max_cost = float(df['cost_of_the_order'].max())
            cost_range = st.slider(
                "Order Value Range ($)",
                min_value=min_cost,
                max_value=max_cost,
                value=(min_cost, max_cost),
                key='cost_filter'
            )

        # Rating filter
        selected_ratings = None
        if 'rating' in df.columns:
            rating_options = sorted(df['rating'].unique().tolist())
            selected_ratings = st.multiselect(
                "Rating",
                options=rating_options,
                default=rating_options,
                key='rating_filter'
            )

        st.markdown("---")
        st.markdown("##### 📊 Quick Stats")
        st.markdown(f"- **Records:** {len(df):,}")
        st.markdown(f"- **Restaurants:** {df['restaurant_name'].nunique() if 'restaurant_name' in df.columns else 'N/A'}")
        st.markdown(f"- **Cuisines:** {df['cuisine_type'].nunique() if 'cuisine_type' in df.columns else 'N/A'}")

        st.markdown("---")
        st.caption("Built with Streamlit, Plotly & Scikit-learn")
        st.caption("© 2026 FoodHub BI Dashboard")

    return page, selected_cuisines, selected_days, cost_range, selected_ratings


# ============================================================================
# FILTER APPLICATION
# ============================================================================
def apply_filters(df, cuisines, days, cost_range, ratings):
    """Apply sidebar filters to the dataframe."""
    filtered = df.copy()

    if cuisines and 'cuisine_type' in filtered.columns:
        filtered = filtered[filtered['cuisine_type'].isin(cuisines)]

    if days and 'day_of_the_week' in filtered.columns:
        filtered = filtered[filtered['day_of_the_week'].isin(days)]

    if cost_range and 'cost_of_the_order' in filtered.columns:
        filtered = filtered[
            (filtered['cost_of_the_order'] >= cost_range[0]) &
            (filtered['cost_of_the_order'] <= cost_range[1])
        ]

    if ratings and 'rating' in filtered.columns:
        filtered = filtered[filtered['rating'].isin(ratings)]

    return filtered


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""

    # --- Load Data ---
    raw_df, data_info = load_data()

    # --- Clean Data ---
    cleaned_df, cleaning_report = clean_data(raw_df)

    # --- Engineer Features ---
    featured_df = engineer_features(cleaned_df)

    # --- Render Sidebar & Get Filters ---
    page, sel_cuisines, sel_days, cost_range, sel_ratings = render_sidebar(featured_df)

    # --- Apply Filters ---
    filtered_df = apply_filters(featured_df, sel_cuisines, sel_days,
                                cost_range, sel_ratings)

    # Show filter warning if data is significantly reduced
    if len(filtered_df) < len(featured_df) * 0.1:
        st.warning(f"⚠️ Filters reduced data to {len(filtered_df)} records "
                   f"({len(filtered_df)/len(featured_df)*100:.1f}% of total). "
                   f"Results may not be representative.")
    elif len(filtered_df) < len(featured_df):
        st.info(f"📊 Showing {len(filtered_df):,} of {len(featured_df):,} records "
                f"({len(filtered_df)/len(featured_df)*100:.1f}%) based on filters.")

    # --- Calculate KPIs ---
    kpis = calculate_kpis(filtered_df)

    # --- Customer Analysis (use full data for segmentation stability) ---
    customer_df, _ = perform_customer_analysis(featured_df)

    # --- AI/ML Analysis ---
    model_output, customer_df_with_risk, all_results = train_risk_model(customer_df)

    # --- Generate Insights ---
    insights = generate_business_insights(filtered_df, kpis, customer_df)

    # --- Generate Recommendations ---
    recommendations = generate_recommendations(
        filtered_df, kpis, customer_df, model_output
    )

    # --- Render Selected Page ---
    if "Executive Overview" in page:
        render_executive_overview(filtered_df, kpis, insights)

    elif "Sales & Restaurant" in page:
        render_sales_analysis(filtered_df, kpis)

    elif "Customer Intelligence" in page:
        render_customer_intelligence(filtered_df, customer_df, kpis)

    elif "Delivery & Operations" in page:
        render_delivery_operations(filtered_df, kpis)

    elif "AI Risk Analysis" in page:
        render_ai_risk_analysis(model_output, customer_df_with_risk)

    elif "Opportunities & Actions" in page:
        render_opportunities_actions(recommendations, kpis, model_output)

    elif "Data Quality" in page:
        render_data_quality(filtered_df, cleaning_report, data_info,
                           kpis, model_output)


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()

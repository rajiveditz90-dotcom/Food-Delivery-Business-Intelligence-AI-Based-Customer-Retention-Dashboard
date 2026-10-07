# Food Delivery Business Intelligence & AI-Based Customer Retention Dashboard

## Project Overview

This project transforms real food-delivery transactional data into a decision-oriented Business Intelligence dashboard with an AI-powered customer risk scoring system.

**RAW DATA → DATA CLEANING → KPI ANALYSIS → TREND ANALYSIS → BUSINESS DRIVERS → CUSTOMER INSIGHTS → AI/ML RISK ANALYSIS → OPPORTUNITIES → RECOMMENDED ACTIONS**

The dashboard is designed to answer:
1. How is the business performing?
2. What is driving revenue and orders?
3. Which restaurants/categories are performing well or poorly?
4. Which customers are valuable?
5. Which customers may be at risk of becoming inactive?
6. What operational issues are affecting performance?
7. What opportunities exist?
8. What should management do next?

## Problem Statement

Food delivery platforms generate massive amounts of transactional data, but the raw records are difficult to use directly for management decisions. This project converts food-delivery order data into KPIs, interactive visualizations, customer intelligence, AI-based risk predictions and evidence-based decision support.

## Objectives

- Clean and validate food delivery order data.
- Calculate business KPIs from available fields.
- Analyze revenue, orders, restaurants, cuisines, customers and operations.
- Segment customers using an RFM-inspired methodology.
- Build an AI/ML customer risk scoring model.
- Identify measurable risks and opportunities.
- Convert findings into Fact → Insight → Risk/Opportunity → Action recommendations.
- Provide an interactive Streamlit dashboard.

## Dataset

**Dataset:** FoodHub Order Analysis Dataset  
**Source:** Public GitHub Repository (MIT Professional Education Data Science Projects)  
**Dataset URL:** https://raw.githubusercontent.com/prneidhardt/Python-Foundations/main/foodhub_order.csv

The dataset contains 1,898 food delivery order records from a NYC-based food aggregator. It contains nine variables: order_id, customer_id, restaurant_name, cuisine_type, cost_of_the_order, day_of_the_week, rating, food_preparation_time and delivery_time.

### Dataset use

The project does **not** use a dataset from the internship masterclasses. FoodHub Order Analysis is a separately sourced public dataset from GitHub.

## Data Cleaning

The application performs the following steps at runtime:

1. Remove duplicate rows.
2. Handle 'Not given' ratings by converting to NaN and creating a numeric rating column.
3. Ensure correct data types for cost and time columns.
4. Remove records with non-positive order cost.
5. Flag cost outliers using the IQR method.
6. Create a total_time feature (food_preparation_time + delivery_time).
7. Create a rating_given boolean flag.

The **Data Quality & Methodology** page displays the actual row counts, missing values, data types and cleaning report generated from the dataset at runtime. Cleaning counts are intentionally not hard-coded in this README.

## KPIs

The dashboard calculates:

- Total Revenue
- Total Orders
- Total Customers
- Average Order Value
- Average Rating
- Repeat Customer Rate
- Total Restaurants
- Average Delivery Time
- Average Preparation Time
- Weekend Order Percentage

Cancellation rate and order status metrics are **not** claimed because the source dataset does not contain order status fields.

## Dashboard Pages

### 1. Executive Overview
Management-level snapshot with KPI cards, revenue by cuisine, weekend vs weekday analysis, order value distribution, rating distribution and auto-generated business insights.

### 2. Sales & Restaurant Analysis
Top restaurants by revenue and volume, cuisine performance, Pareto (80/20) analysis, order value segmentation and key business driver identification.

### 3. Customer Intelligence
RFM-inspired segmentation (Champions, Loyal, At Risk, etc.), customer value tiers, orders-per-customer distribution, cuisine exploration vs spending analysis and top customer identification.

### 4. Delivery & Operations
Preparation and delivery time distributions, cuisine-wise operational performance, rating vs delivery speed analysis, weekend vs weekday operations and operational bottleneck identification.

### 5. AI Risk Analysis
AI-Assisted Customer Risk Scoring using Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting). Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC), feature importance, risk distribution and high-risk customer identification.

### 6. Opportunities & Actions
Impact vs Urgency priority matrix, prioritized action plan organized by DO NOW / PLAN / DELEGATE / CONSIDER quadrants, with each recommendation following Fact → Insight → Risk/Opportunity → Action structure.

### 7. Data Quality & Methodology
Dataset information, cleaning report, missing values analysis, feature engineering documentation, KPI definitions, ML methodology, known limitations and data sample.

## AI/ML Methodology

The dataset does not contain date/timestamp information, so traditional churn prediction (based on a time-based churn window) is not possible. Instead, an **AI-Assisted Customer Risk Scoring** system is implemented:

- **Risk label definition:** Customers are labeled at-risk based on low order frequency, low spend, low engagement and poor service experience.
- **Models evaluated:** Logistic Regression, Random Forest, Gradient Boosting.
- **Best model selected:** Based on F1-Score.
- **Output:** Risk probability score per customer, categorized into High / Medium / Low Risk.

The model is clearly labeled as a behavioral risk scoring system, not a validated churn prediction model.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## How to Run

```bash
pip install -r requirements.txt
streamlit run Rajiv_FoodDeliveryBI.py
```

The application downloads the dataset automatically if the local file is not present.

## Project Structure

```
.gitignore
.streamlit/config.toml
README.md
Rajiv_FoodDeliveryBI.py
Rajiv_ProjectReport.docx
requirements.txt
```

## Limitations

- No date/timestamp data — cannot analyze temporal trends or define true churn windows.
- No geographic/location data — cannot perform location-based analysis.
- No order status data — cannot analyze cancellations or failures.
- No payment method data — cannot analyze payment preferences.
- No discount/promotion data — cannot assess promotional impact.
- Ratings have a significant 'Not given' percentage — satisfaction metrics may be biased.
- Dataset size (~1,898 orders) is relatively small for production ML models.
- Risk scoring is behavioral, not validated temporal churn prediction.

## Future Scope

- Integration with real-time streaming data.
- Location-based analysis once geographic data is available.
- Recommendation engine for cross-selling cuisines.
- A/B testing framework for promotional campaigns.
- Time-series churn prediction with timestamped order data.

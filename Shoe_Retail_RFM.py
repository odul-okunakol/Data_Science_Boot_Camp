###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################
# The shoe retail company wants to segment its customers and determine marketing strategies accordingly.
# To achieve this, customer behaviors will be analyzed and grouped based on their patterns.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information derived from past shopping behaviors of customers
# who made their last purchases between 2020 and 2021 via the OmniChannel (both online and offline).

# master_id: Unique customer ID
# order_channel: The platform/channel used for the purchase (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel: The channel where the most recent purchase was made
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's most recent purchase
# last_order_date_online: The date of the customer’s last online purchase
# last_order_date_offline: The date of the customer’s last offline purchase
# order_num_total_ever_online: Total number of purchases made by the customer online
# order_num_total_ever_offline: Total number of purchases made by the customer offline
# customer_value_total_ever_offline: Total amount paid by the customer for offline purchases
# customer_value_total_ever_online: Total amount paid by the customer for online purchases
# interested_in_categories_12: List of categories the customer shopped from in the last 12 months

###############################################################
# Data Preparation and Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("CRM_Analytics/data_sets/data_20k.csv")
df = df_.copy()

# 2. Examine the following in the dataset:
        # a. First 10 observations
        # b. Variable names
        # c. Shape
        # d. Descriptive statistics
        # e. Missing values
        # f. Variable types

# a
df.head(10)
# b
df.info()
# df.columns, list(df.columns)
# c
df.describe().T
# d
df.isnull().sum()
# e
df.info()

# 3. Omnichannel refers to customers who shop from both online and offline platforms.
# Create new variables for total number of purchases and total spending for each customer.

df['total_order'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_purchase'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.info()

# 4. Examine variable types and convert date-related columns to datetime format.

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df.info()

# 5. Analyze distribution of customer counts, total products purchased, and total spending across shopping channels.

df.groupby("order_channel").agg(
    customer_count=("master_id", "nunique"),
    total_products=("total_order", "sum"),
    total_spent=("total_purchase", "sum")
)

# 6. List the top 10 customers generating the highest revenue.

top_customers = (
    df.groupby("master_id")
    .agg({"total_purchase": "sum"})
    .sort_values(by="total_purchase", ascending=False)
    .head(10)
)

# 7. List the top 10 customers with the highest number of orders.

top_customers1 = (
    df.groupby("master_id")
    .agg({"total_order": "sum"})
    .sort_values(by="total_order", ascending=False)
    .head(10)
)

# 8. Turn the data preparation process into a function.

###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Set analysis date to 2 days after the last purchase in the dataset

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)

# Create a new RFM dataframe containing customer_id, recency, frequency, and monetary

rfm_flo = df.groupby('master_id').agg({
    'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
    'total_order': lambda total_order: total_order.nunique(),
    'total_purchase': lambda total_purchase: total_purchase.sum()
})
rfm_flo.head()

rfm_flo.columns = ['recency', 'frequency', 'monetary']

rfm_flo.describe().T

rfm = rfm_flo[rfm_flo["monetary"] > 0]
rfm.shape

###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

# Convert Recency, Frequency, and Monetary metrics into scores between 1-5 using qcut
# Save the scores as recency_score, frequency_score, and monetary_score

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine recency_score and frequency_score into a single variable RF_SCORE

rfm["RF_SCORE"] = (
    rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
)

rfm.describe().T
rfm[rfm["RF_SCORE"] == "55"]
rfm[rfm["RF_SCORE"] == "11"]

###############################################################
# TASK 4: Defining Segments Based on RF Scores
###############################################################

# To make the RFM scores more interpretable, define segments and map RF_SCORE to them using seg_map

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'can’t_lose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "can’t_lose"].head()
rfm[rfm["segment"] == "can’t_lose"].index

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
# new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
new_df.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")

###############################################################
# TASK 5: Time to Act!
###############################################################

# 1. Examine the mean recency, frequency, and monetary values of the segments.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# 2. Using RFM analysis, identify customers for 2 specific cases and save their IDs to CSV.

# a. The shoe retail company is introducing a new women's shoe brand.
# The brand's prices are higher than the general preference of customers.
# Therefore, they want to contact loyal customers who have shopped from the women’s category.
# Save these customer IDs to a file named new_brand_target_customer_ids.csv

loyal_female_customers = rfm[
    (rfm["segment"].isin(["champions", "loyal_customers"])) &
    (df["interested_in_categories_12"] == "KADIN")
]

# b. The company is planning a 40% discount on men's and children's products.
# The target is to reach previously good customers who haven’t shopped for a long time and new customers interested in these categories.
# Save the customer IDs to a file named discount_target_customer_ids.csv




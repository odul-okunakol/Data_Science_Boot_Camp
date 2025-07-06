##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# Data Preparation
###############################################################

# 1. Read the OmniChannel.csv file and create a copy of the dataframe.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv("CRM_Analytics/data_sets/data_20k.csv")
df = df_.copy()

# 2. Define the functions `outlier_thresholds` and `replace_with_thresholds` to suppress outliers.
# Note: While calculating CLTV, the frequency values must be integers. Therefore, use round() for lower and upper limits.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. If there are any outliers in the following variables,
# suppress them:
# "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"

# Example (apply to each manually if not shown here):
# replace_with_thresholds(df, "order_num_total_ever_online")

# 4. Omnichannel indicates that customers shop from both online and offline platforms.
# Create new variables for the total number of purchases and total expenditure per customer.

df['total_order'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_purchase'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.info()

# 5. Check the data types. Convert the date-related variables to datetime format.

df["first_order_date"] = df["first_order_date"].apply(pd.to_datetime)
df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)
df["last_order_date_online"] = df["last_order_date_online"].apply(pd.to_datetime)
df["last_order_date_offline"] = df["last_order_date_offline"].apply(pd.to_datetime)

###############################################################
# Creating CLTV Data Structure
###############################################################


# 1. Take 2 days after the last purchase in the dataset as the analysis date.
df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)

# 2. Create a new CLTV dataframe that includes the following:
# customer_id, recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg

cltv_df = df.groupby('master_id').agg({
    'last_order_date': [
        lambda last_order_date: (last_order_date.max() - last_order_date.min()).days,
        lambda last_order_date: (today_date - last_order_date.min()).days
    ],
    'total_order': lambda total_order: total_order.nunique(),
    'total_purchase': lambda total_purchase: total_purchase.sum()
})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

# Filter out customers with frequency <= 1
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# Convert recency and T into weeks
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

###############################################################
# BG/NBD and Gamma-Gamma Model Fitting & 6-Month CLTV Calculation
###############################################################

# 1. Fit the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# Estimate expected purchases in 3 months and add as `exp_sales_3_month` to cltv dataframe
bgf.conditional_expected_number_of_purchases_up_to_time(
    3,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
).sort_values(ascending=False).head(10)

bgf.predict(
    3,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(
    1,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
)

# Estimate expected purchases in 6 months and add as `exp_sales_6_month` to cltv dataframe
bgf.conditional_expected_number_of_purchases_up_to_time(
    6,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
).sort_values(ascending=False).head(10)

bgf.predict(
    6,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(
    1,
    cltv_df['frequency'],
    cltv_df['recency'],
    cltv_df['T']
)






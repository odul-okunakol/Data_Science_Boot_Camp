##################################################
# Revenue Prediction with Rule-Based Classification
##################################################

# Business Problem:
# A gaming company wants to define new level-based customer personas using certain user characteristics.
# Based on these personas, the company aims to create segments and estimate the potential revenue
# that a new user might bring.

# For example: What would be the average revenue from a 25-year-old male iOS user from Turkey?

##################################################
# Dataset Description:
##################################################
# The dataset (persona.csv) contains sales transactions of an international gaming company.
# It includes the prices of products sold and demographic details of the customers who purchased them.
# Note: The table is not deduplicated – a customer with the same profile may appear more than once.

# Features:
# PRICE   : Amount paid by the customer
# SOURCE  : Device type (iOS or Android)
# SEX     : Customer's gender
# COUNTRY : Customer's country
# AGE     : Customer's age

#################### Initial Data Example ####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

#################### Final Output Example ####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

##################################################
# PROJECT TASKS
##################################################

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

##################################################
# TASK 1: Basic Exploration
##################################################

# Load the dataset
df = pd.read_csv("persona.csv")
print(df.head())

# Number of unique SOURCE types and their frequencies
print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())

# Number of unique PRICE values
print(df["PRICE"].nunique())

# Frequency of each PRICE
print(df["PRICE"].value_counts())

# Sales count per COUNTRY
print(df["COUNTRY"].value_counts())

# Total revenue per COUNTRY
print(df.groupby('COUNTRY')['PRICE'].sum())

# Sales count per SOURCE type
print(df.groupby('SOURCE')['PRICE'].value_counts())

# Average PRICE per COUNTRY
print(df.groupby('COUNTRY')['PRICE'].mean())

# Average PRICE per SOURCE
print(df.groupby('SOURCE')['PRICE'].mean())

# Average PRICE for each COUNTRY-SOURCE pair
print(df.groupby(['SOURCE', 'COUNTRY'])['PRICE'].mean())

##################################################
# TASK 2: Average revenue based on COUNTRY, SOURCE, SEX, and AGE
##################################################

agg_df = df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE'])['PRICE'].mean()

##################################################
# TASK 3: Sort the output by PRICE in descending order
##################################################

agg_df = agg_df.sort_values(ascending=False)
print(agg_df)

##################################################
# TASK 4: Reset index to convert index names into columns
##################################################

agg_df = agg_df.reset_index()
print(agg_df)

##################################################
# TASK 5: Convert AGE into a categorical variable
##################################################

age_bins = [0, 18, 23, 30, 40, 70]
age_labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df['AGE_CATEGORY'] = pd.cut(agg_df['AGE'], bins=age_bins, labels=age_labels)
print(agg_df)

##################################################
# TASK 6: Create new level-based customer profiles
##################################################

agg_df['customers_level_based'] = (
    agg_df['COUNTRY'].str.upper() + "_" +
    agg_df['SOURCE'].str.upper() + "_" +
    agg_df['SEX'].str.upper() + "_" +
    agg_df['AGE_CATEGORY'].astype(str)
)

# Group by the new profile and calculate mean PRICE
agg_df = agg_df.groupby('customers_level_based').agg({'PRICE': 'mean'}).reset_index()
print(agg_df)

##################################################
# TASK 7: Segment the new customers based on PRICE
##################################################

agg_df['SEGMENT'] = pd.qcut(agg_df['PRICE'], q=4, labels=['D', 'C', 'B', 'A'])

# Segment summary
segment_summary = agg_df.groupby('SEGMENT').agg({'PRICE': ['mean', 'max', 'min', 'count']})
print("Segment Summary:")
print(segment_summary)

##################################################
# TASK 8: Estimate revenue and segment for new customers
##################################################

# Example 1: 33-year-old Turkish female using Android
new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])

# Example 2: 35-year-old French female using iOS
new_user = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])



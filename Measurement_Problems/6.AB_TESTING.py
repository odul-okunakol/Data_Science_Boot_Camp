
#####################################################
# A/B Testing: Comparing Conversion of Bidding Strategies
#####################################################

#####################################################
# Dataset Story
#####################################################

# This dataset contains website data for a company, including metrics such as the number of ads viewed and clicked by users,
# as well as the revenue generated from those clicks. The dataset consists of two groups: Control and Test.
# These are stored in different sheets of the ab_testing.xlsx file.
# The Control group used the Maximum Bidding strategy, while the Test group used the Average Bidding strategy.

# impression: Number of ad impressions
# Click: Number of clicks on the ad
# Purchase: Number of purchases after clicking the ad
# Earning: Revenue generated from purchases

#####################################################
# Project Tasks
#####################################################

#####################################################
# A/B Testing (Independent Two-Sample T-Test)
#####################################################

# 1. Define the hypotheses
# 2. Check assumptions:
#    - 1. Normality assumption (Shapiro-Wilk test)
#    - 2. Homogeneity of variances (Levene's test)
# 3. Apply the hypothesis test:
#    - If assumptions are met: independent two-sample t-test
#    - If assumptions are not met: Mann-Whitney U test
# 4. Interpret the results based on the p-value

# Note:
# - If normality is not met, directly proceed to step 2 (non-parametric).
# - If variance homogeneity is not met, use the appropriate argument in the t-test.
# - It is recommended to check and handle outliers before normality testing.

#####################################################
# TASK 1: Data Preparation and Exploration
#####################################################
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from pandas import read_csv, read_excel
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Step 1: Load the dataset named ab_testing.xlsx, which includes both control and test groups.
# Assign control and test group data to separate variables.

df = pd.ExcelFile("3.measurement/datasets/ab_testing.xlsx")

df_control = pd.read_excel(df, sheet_name="Control Group")  # Control group (Maximum Bidding)
df_test = pd.read_excel(df, sheet_name="Test Group")        # Test group (Average Bidding)

# Step 2: Analyze the control and test group data
df_control.describe().T
df_test.describe().T

def compare_dfs(df1, df2, df1_name="Control Group", df2_name="Test Group"):
    # Function to return a side-by-side statistical summary of two DataFrames
    desc1 = df1.describe().T
    desc2 = df2.describe().T
    comparison_df = pd.concat([desc1, desc2], axis=1, keys=[df1_name, df2_name])
    return comparison_df

comparison_result = compare_dfs(df_control, df_test)
print(comparison_result)

# Step 3: Use concat method to merge control and test group data

# Add a 'Group' column to each DataFrame before merging
df_control["Group"] = "Control"
df_test["Group"] = "Test"

# Merge the two groups
df_combined = pd.concat([df_control, df_test], ignore_index=True)

print(df_combined.head())
df_combined.info()

#####################################################
# TASK 2: Defining the Hypotheses of the A/B Test
#####################################################

# Step 1: Define the hypotheses

# H0: M1 = M2 (There is no difference in purchase means between control and test groups)
# H1: M1 ≠ M2 (There is a difference in purchase means between control and test groups)

# Step 2: Analyze the mean purchase values for control and test groups
df_combined.groupby("Group").agg({"Purchase": "mean"})

#####################################################
# Conducting the Hypothesis Test
#####################################################

# A/B Testing (Independent Two-Sample T-Test)

# Before performing the test, check assumptions:
# Normality and Homogeneity of Variances using Shapiro-Wilk and Levene’s tests

# H0: Data is normally distributed
# H1: Data is not normally distributed

test_stat, pvalue = shapiro(df_combined.loc[df_combined["Group"] == "Control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_combined.loc[df_combined["Group"] == "Test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Step 2: Based on normality and homogeneity results, choose the appropriate test
# H0: Variances are homogeneous
# H1: Variances are not homogeneous

test_stat, pvalue = ttest_ind(df_combined.loc[df_combined["Group"] == "Control", "Purchase"],
                              df_combined.loc[df_combined["Group"] == "Test", "Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Step 3: Based on the p-value, interpret whether there is a statistically significant difference
# between the mean purchases of the control and test groups.

# Conclusion: Based on the p-value, there is no statistically significant difference
# between the average purchases of the control and test groups.

##############################################################
# Final Analysis
##############################################################

# Step 1: State which test was used and why.

# Since assumptions were met, an independent two-sample t-test (parametric test) was used.
# We used the A/B testing methodology with an independent t-test because:
# - The normality assumption was met
# - Variance homogeneity was confirmed

# Step 2: Make recommendations to the client based on the results.

# According to the test results, there is no statistically significant difference between
# the strategies: Average Bidding (Test Group) and Maximum Bidding (Control Group).

# Therefore, instead of changing the current strategy, the company is advised to:
# - Conduct further testing, or
# - Explore new advertising models.

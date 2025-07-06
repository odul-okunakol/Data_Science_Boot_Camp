
##################################################
# List Comprehensions
##################################################

###############################################
# TASK 1: Using list comprehension, convert the names of all numeric variables in the car_crashes dataset to uppercase and prefix them with "NUM".
###############################################

# Expected Output:
# ['NUM_TOTAL',
#  'NUM_SPEEDING',
#  'NUM_ALCOHOL',
#  'NUM_NOT_DISTRACTED',
#  'NUM_NO_PREVIOUS',
#  'NUM_INS_PREMIUM',
#  'NUM_INS_LOSSES',
#  'ABBREV']

# Notes:
# Non-numeric column names should also be uppercased.
# Only one list comprehension should be used.

import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()

# Solution:
df.columns = ["NUM_" + col.upper() if col in df.select_dtypes(include=["number"]).columns else col.upper() for col in df.columns]
df.columns

###############################################
# TASK 2: Using list comprehension, add the suffix "FLAG" to the names of variables that do NOT contain "no" in their names.
###############################################

# Notes:
# All variable names must be uppercase.
# Only one list comprehension should be used.

# Expected Output:
# ['TOTAL_FLAG',
#  'SPEEDING_FLAG',
#  'ALCOHOL_FLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM_FLAG',
#  'INS_LOSSES_FLAG',
#  'ABBREV_FLAG']

df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() + "_FLAG" if "NO" not in col.upper() else col.upper() for col in df.columns]
df.columns

###############################################
# TASK 3: Using list comprehension, select the variable names that are NOT in the list below, and create a new dataframe with only those variables.
###############################################

og_list = ["abbrev", "no_previous"]

# Solution:
df = sns.load_dataset("car_crashes")
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df

# Notes:
# First, create a new list named new_cols using list comprehension.
# Then, create a new dataframe called new_df using df[new_cols].

# Expected Output:
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0 18.800     7.332    5.640          18.048      784.550     145.080
# 1 18.100     7.421    4.525          16.290     1053.480     133.930
# 2 18.600     6.510    5.208          15.624      899.470     110.350
# 3 22.400     4.032    5.824          21.056      827.340     142.390
# 4 12.000     4.200    3.360          10.920      878.410     165.630








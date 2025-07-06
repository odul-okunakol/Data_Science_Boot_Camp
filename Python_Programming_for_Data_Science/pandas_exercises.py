##################################################
# Pandas Exercises
##################################################

import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# TASK 1: Load the Titanic dataset from the Seaborn library.
#########################################
df = sns.load_dataset("titanic")
df.head()

#########################################
# TASK 2: Count the number of female and male passengers.
#########################################
df["sex"].value_counts()

#########################################
# TASK 3: Find the number of unique values for each column.
#########################################
df.nunique()

#########################################
# TASK 4: Find the unique values of the "pclass" variable.
#########################################
df["pclass"].unique()

#########################################
# TASK 5: Find the number of unique values for "pclass" and "parch".
#########################################
df["pclass"].nunique()
df["parch"].nunique()

#########################################
# TASK 6: Check the data type of "embarked", then convert it to "category" and recheck.
#########################################
print(df['embarked'].dtype)
df['embarked'] = df['embarked'].astype('category')
print(df['embarked'].dtype)

#########################################
# TASK 7: Display all records where "embarked" is "C".
#########################################
embarked_data = df[df['embarked'] == 'C']
print(embarked_data)

#########################################
# TASK 8: Display all records where "embarked" is NOT "S".
#########################################
embarked_data_S = df[df['embarked'] != 'S']
print(embarked_data_S)

#########################################
# TASK 9: Show all female passengers under 30 years old.
#########################################
filtered_1 = df[(df['age'] < 30) & (df['sex'] == 'female')]
filtered_1

#########################################
# TASK 10: Show passengers who paid more than 500 or are older than 70.
#########################################
filtered_2 = df[(df['fare'] > 500) | (df['age'] > 70)]
filtered_2

#########################################
# TASK 11: Find the total number of missing values in the dataset.
#########################################
total_missing = df.isnull().sum().sum()
total_missing

#########################################
# TASK 12: Drop the "who" column from the dataframe.
#########################################
df_dropped = df.drop(columns=['who'])
df_dropped

#########################################
# TASK 13: Fill missing values in "deck" with its mode (most frequent value).
#########################################
# Method 1
mode_value = df['deck'].value_counts().idxmax()
df.loc[df['deck'].isnull(), 'deck'] = mode_value

# Method 2 (simpler)
deck_mode = df['deck'].mode()[0]
df['deck'] = df['deck'].fillna(deck_mode)
df[df['deck'].isnull()]

#########################################
# TASK 14: Fill missing values in "age" with the median age.
#########################################
age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)
df['age'].isnull()

#########################################
# TASK 15: Calculate sum, count, and mean of "survived" grouped by "pclass" and "sex".
#########################################
result_survived = df.groupby(['pclass', 'sex'])['survived'].agg(['sum', 'count', 'mean'])
result_survived

#########################################
# TASK 16: Create a function that assigns 1 if age < 30, else 0. Use it to create a new column "age_flag".
#########################################
df['age_flag'] = df['age'].apply(lambda age: 1 if age < 30 else 0)
df['age_flag']

#########################################
# TASK 17: Load the "tips" dataset from the Seaborn library.
#########################################
df = sns.load_dataset("tips")
df.info()

#########################################
# TASK 18: Calculate sum, min, max, and mean of "total_bill" grouped by "time".
#########################################
result_time = df.groupby(['time'], observed=True)['total_bill'].agg(['sum', 'min', 'max', 'mean'])
result_time

#########################################
# TASK 19: Calculate sum, min, max, and mean of "total_bill" grouped by "time" and "day".
#########################################
result_timedays = df.groupby(['time', 'day'], observed=True)['total_bill'].agg(['sum', 'min', 'max', 'mean'])
result_timedays

#########################################
# TASK 20: For female customers during Lunch, find sum, min, max, and mean of "total_bill" and "tip" by "day".
#########################################
result_lunchf = df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')].groupby('day', observed=True)[['total_bill', 'tip']].agg(['sum', 'min', 'max', 'mean'])
result_lunchf

#########################################
# TASK 21: What is the average total_bill for orders where size < 3 and total_bill > 10?
#########################################
filtered_df = df[(df['size'] < 3) & (df['total_bill'] > 10)]
mean_total_bill = filtered_df['total_bill'].mean()
mean_total_bill

#########################################
# TASK 22: Create a new column "total_bill_tip_sum" as the sum of "total_bill" and "tip".
#########################################
df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
df['total_bill_tip_sum']

#########################################
# TASK 23: Sort the dataset by "total_bill_tip_sum" in descending order and create a new dataframe with the top 30 records.
#########################################
df_sorted = df.sort_values(by='total_bill_tip_sum', ascending=False).head(30)
df_sorted





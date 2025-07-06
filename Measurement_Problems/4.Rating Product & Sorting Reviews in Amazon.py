###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Dataset Story
###################################################

# This dataset contains Amazon product data, including product categories and various metadata.
# It includes user ratings and reviews of the most reviewed product in the Electronics category.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: Reviewer’s Name
# helpful: Helpfulness rating
# reviewText: Review content
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review timestamp (UNIX format)
# reviewTime: Raw review time
# day_diff: Number of days since the review
# helpful_yes: Number of times the review was marked helpful
# total_vote: Total number of votes received by the review

###################################################
# Calculate the Average Rating Based on Recent Reviews and Compare with Existing Average Rating
###################################################

# In the provided dataset, users gave ratings and posted reviews for a product.
# The goal in this task is to calculate a time-weighted rating based on the recency of reviews.
# Then, compare the new weighted rating with the original average rating.

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Load the Dataset and Calculate the Product's Average Rating
###################################################
df = pd.read_csv("3.measurement/datasets/amazon_review.csv")
df.info()
df.head()
df.shape
df["overall"].mean()

###################################################
# Calculate the Time-Based Weighted Average Rating
###################################################

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.info()
df["day_diff"].max()
df["day_diff"].describe()

def time_based_weighted_average(df, w1=30, w2=26, w3=22, w4=22):
    return df.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           df.loc[(df["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

###################################################
# Compare and Interpret the Averages of Each Time Interval in the Weighted Rating
###################################################

def user_based_weighted_average(df, w1=22, w2=24, w3=26, w4=28):
    return df.loc[df["day_diff"] <= 10, "overall"].mean() * w1 / 100 + \
           df.loc[(df["day_diff"] > 10) & (df["day_diff"] <= 45), "overall"].mean() * w2 / 100 + \
           df.loc[(df["day_diff"] > 45) & (df["day_diff"] <= 75), "overall"].mean() * w3 / 100 + \
           df.loc[(df["day_diff"] > 75), "overall"].mean() * w4 / 100

user_based_weighted_average(df, 20, 24, 26, 30)

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)
course_weighted_rating(df, time_w=45, user_w=55)

###################################################
#  Identify the Top 20 Reviews to Display on the Product Detail Page
###################################################

###################################################
# Create the 'helpful_no' Variable
###################################################

# Note:
# total_vote is the sum of up and down votes for a review.
# 'helpful_yes' indicates upvotes.
# There is no 'helpful_no' variable in the dataset — it must be derived.

df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.info()

###################################################
#Calculate score_pos_neg_diff, score_average_rating, and wilson_lower_bound Scores and Add to DataFrame
###################################################
import numpy as np
from scipy.stats import norm

# 1. Score: Difference Between Positive and Negative Votes
df['score_pos_neg_diff'] = df['helpful_yes'] - df['helpful_no']

# 2. Score: Average Rating (avoid division by zero)
df['score_average_rating'] = np.where(df['total_vote'] > 0, df['helpful_yes'] / df['total_vote'], 0)

# 3. Score: Wilson Lower Bound
def wilson_lower_bound(helpful_yes, total_vote, confidence=0.95):
    if total_vote == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = helpful_yes / total_vote
    return (phat + z**2 / (2 * total_vote) - z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total_vote)) / total_vote)) / (1 + z**2 / total_vote)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['total_vote']), axis=1)

# (optional) If working with a second DataFrame:
# df_votes["wilson_lower_bound"] = df_votes.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["total_vote"]), axis=1)

###################################################
#Select and Interpret the Top 20 Reviews
###################################################

# Calculate and sort based on Wilson Lower Bound score
top_20_reviews = df.sort_values(by='wilson_lower_bound', ascending=False).head(20)

# Display results
print("\n🔹 Top 20 Reviews Based on Wilson Lower Bound Score 🔹\n")
print(top_20_reviews[['total_vote', 'helpful_yes', 'helpful_no',
                      'score_pos_neg_diff', 'score_average_rating',
                      'wilson_lower_bound']])





#########################
# Business Problem
#########################

# Armut is Turkey’s largest online service platform that connects service providers with customers.
# It enables easy access to services like cleaning, renovation, and moving with just a few clicks
# via computer or smartphone.
# Using a dataset that includes users and the services/categories they received,
# a product recommendation system is to be built using Association Rule Learning.


#########################
# Dataset Information
#########################

# The dataset consists of services received by customers and their corresponding categories.
# Each service entry includes the date and time it was purchased.

# UserId: Unique ID of the customer
# ServiceId: Anonymized service ID under each category (e.g., sofa cleaning under cleaning category)
#            A single ServiceId can exist under different categories and refer to different services.
#            (e.g., CategoryId 7 & ServiceId 4 = radiator cleaning,
#                   CategoryId 2 & ServiceId 4 = furniture assembly)
# CategoryId: Anonymized category ID (e.g., cleaning, moving, renovation)
# CreateDate: Date when the service was purchased


#########################
# Data preparition
#########################
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_csv("Recommdation_Systems/datasets/armut_data.csv")

df = df_.copy()
df.head()

# Each ServiceID represents a different service depending on the associated CategoryID.
# Create a new variable to represent the services by combining ServiceID and CategoryID with an underscore ("_").


df['Hizmet'] = df['ServiceId'].astype(str) + "_" + df['CategoryId'].astype(str)

# The dataset consists of the date and time when the services were received;
# there is no predefined basket (e.g., invoice) structure.
# To apply Association Rule Learning, a basket (e.g., invoice) definition needs to be created.
# Here, the basket is defined as the services received by each customer on a monthly basis.
# For example, services 9_4 and 46_4 received by customer with ID 7256 in August 2017 represent one basket;
# services 9_4 and 38_4 received by the same customer in October 2017 represent another basket.
# Each basket should be identified with a unique ID.
# To do this, first create a new date variable that includes only the year and month.
# Then, create a new variable named 'ID' by combining the UserID and the new date variable using an underscore ("_").

df['CreateDate'] = pd.to_datetime(df['CreateDate'])

#Adding new columns
df['Yil'] = df['CreateDate'].dt.year
df['Ay'] = df['CreateDate'].dt.month
df['Gün'] = df['CreateDate'].dt.day

df['New_Date'] = df['Yil'].astype(str) + "_" + df['Ay'].astype(str)

df['SepetId'] = df['UserId'].astype(str) + "_" + df['New_Date'].astype(str)
#########################
#Generate Association Rules
#########################

# Create a basket-service pivot table as shown below.


# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


df.set_index('SepetId', inplace=True)



pivot_df = df.pivot_table(index='Hizmet',
                          columns='Kategori', values='Satış', aggfunc='sum')


# Step 2: Generate association rules.
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Extract association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(rules.head())




frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

check_id(df_fr, 21086)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

# Using the arl_recommender function, make a service recommendation for a user who last received the service 2_0.

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)



import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

#Exploratory Data Analysis
#Read the train and test datasets and combine them. Proceed using the merged data.
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

test['SalePrice'] = None

df = pd.concat([train, test], axis=0, ignore_index=True)

print(df.shape)
print(df.head())

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=2)

df.head()
df.info()

## Identify the numerical and categorical variables.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

## Make the necessary adjustments (such as variables with type errors).
print(df.dtypes)

def duzenle_tipler(df):
    # 1) Fix variables that are of type 'object' but should be numeric
    if df['SalePrice'].dtype == 'object':
        df['SalePrice'] = pd.to_numeric(df['SalePrice'], errors='coerce')
        print("SalePrice kolonunu sayısal tipe dönüştürdüm.")

    # 2) Convert columns to categorical type if they are more appropriate as categorical variables
    kategorik_olsun = ['MSSubClass', 'MoSold', 'YrSold']
    for col in kategorik_olsun:
        if col in df.columns:
            df[col] = df[col].astype(str)
            print(f"{col} kolonunu kategorik tipe (string) çevirdim.")

    # 3) Print the data types
    print("\nGüncel kolon tipleri:")
    print(df.dtypes)

    # 4) Check for missing data
    eksikler = df.isnull().sum()
    eksikler = eksikler[eksikler > 0]
    if len(eksikler) > 0:
        print("\nEksik veri olan kolonlar:")
        print(eksikler.sort_values(ascending=False))
    else:
        print("\nEksik veri bulunmuyor.")

    return df


train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
test['SalePrice'] = None
df = pd.concat([train, test], axis=0, ignore_index=True)

df = duzenle_tipler(df)


## Observe the distribution of numerical and categorical variables in the data.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


#or
def cat_summary_l(dataframe, cat_cols, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary_l(df, cat_cols)

for col in cat_cols:
    cat_summary(df,col,plot=True)


##################################
## Analysis of numerical variables
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


 ## Analyze the relationship between categorical variables and the target variable.


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# # Make sure to convert 'SalePrice' to numeric
df['SalePrice'] = pd.to_numeric(df['SalePrice'], errors='coerce')

# # Filter rows where the 'SalePrice' value is not missing
df_filtered = df[df['SalePrice'].notnull()]

# # Select numerical columns
num_cols = [col for col in df.columns if df[col].dtype in [int, float] and col != "SalePrice"]

# # Call the function
for col in num_cols:
    target_summary_with_num(df_filtered, "SalePrice", col)


# Check if there are any outliers.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def count_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].shape[0]

for col in num_cols:
    count = count_outliers(df, col)
    if count > 0:
        print(f"{col} değişkeninde {count} adet aykırı değer var.")
    else:
        print(f"{col} değişkeninde aykırı değer bulunmuyor.")

import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure()
    df.boxplot(column=col)
    plt.title(col)
    plt.show()

## Check if there are any missing values.
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

# Feature Engineering

# Perform the necessary steps for missing and outlier observations.

# Filling missing values:
# Fill missing numerical values with the median
zero_columns = [col for col in df.columns if df[col].isnull().sum() > 0 and df[col].dtype in [int, float]]
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

## Fill missing categorical values with the mode
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Check: Are there any missing values left?
print("Eksik değerler:")
print(df.isnull().sum())

# Check and replace outlier values:
for col in num_cols:
    if check_outlier(df, col):
        print(f"{col} değişkeninde aykırı değer bulundu ve düzeltildi.")
        replace_with_thresholds(df, col)
    else:
        print(f"{col} değişkeninde aykırı değer bulunmadı.")

# Final check: Are there any outlier values left?
for col in num_cols:
    if check_outlier(df, col):
        print(f"{col} değişkeninde hala aykırı değer var.")


# Apply Rare Encoder

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# Rare encoder
#############################################
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)
# Create new variables
def create_house_prices_features(df):
    df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (df['FullBath'] + df['HalfBath'] * 0.5 +
                       df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df['GarageAge'] = df['GarageAge'].fillna(0)
    df['QualitySF'] = df['OverallQual'] * df['TotalHouseSF']
    df['SF_per_room'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 0.01)
    print("House Prices veri setine özel yeni değişkenler eklendi.")
    return df


df = create_house_prices_features(df)


print(df.columns.tolist())


def create_house_prices_features_from_encoded(df):
    # Total living area of the house
    df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Find the total BsmtFullBath from the one-hot columns
    bsmt_fullbath_cols = [col for col in df.columns if col.startswith('BsmtFullBath_')]
    bsmt_fullbath_total = sum([float(col.split('_')[-1]) * df[col] for col in bsmt_fullbath_cols])

    bsmt_halfbath_cols = [col for col in df.columns if col.startswith('BsmtHalfBath_')]
    bsmt_halfbath_total = sum([float(col.split('_')[-1]) * df[col] for col in bsmt_halfbath_cols])

    fullbath_cols = [col for col in df.columns if col.startswith('FullBath_')]
    fullbath_total = sum([int(col.split('_')[-1]) * df[col] for col in fullbath_cols])

    halfbath_cols = [col for col in df.columns if col.startswith('HalfBath_')]
    halfbath_total = sum([int(col.split('_')[-1]) * df[col] for col in halfbath_cols])

    df['TotalBath'] = fullbath_total + halfbath_total * 0.5 + bsmt_fullbath_total + bsmt_halfbath_total * 0.5

    # Age of the house
    df['HouseAge'] = df['YrSold_2010'] * 2010 + df['YrSold_2009'] * 2009 + df['YrSold_2008'] * 2008 + df[
        'YrSold_2007'] * 2007
    df['HouseAge'] = df['HouseAge'] - df['YearBuilt']

    # Quality × area
    df['QualitySF'] = df['OverallQual'] * df['TotalHouseSF']

    # Area per room
    df['SF_per_room'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 0.01)

    print("Encoded kolonlardan yeni değişkenler başarıyla oluşturuldu.")
    return df

df = create_house_prices_features_from_encoded(df)

# Perform encoding operations.

# Separate variables according to their types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One-Hot Encoding operation
# Updating the cat_cols list
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["SalePrice"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
from sklearn.preprocessing import LabelEncoder

# 1. Label Encoding for binary columns
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#2. Update the categorical columns
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["SalePrice"]]

# 3. One-Hot Encoding for remaining categorical cols
df = one_hot_encoder(df, cat_cols, drop_first=True)

# 4. Control
print(f"Final shape: {df.shape}")
print(f"Encoding işlemi tamamlandı.")

# Model Building Step
# Split the data into train and test sets.
# (Rows where the SalePrice variable is missing are the test data.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

X = train_df.drop(["SalePrice", "Id"], axis=1)
y = train_df["SalePrice"]


# Build a model with the training data and evaluate the model performance.
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Split for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

# Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction and error calculation
y_pred = rf_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE (SalePrice orijinal): {rmse}")

# Build a model by applying log transformation to the target variable and observe the results.
# Don’t forget to take the inverse of the log.
# log transformation

y_log = np.log1p(y)

X_train_log, X_valid_log, y_train_log, y_valid_log = train_test_split(X, y_log, test_size=0.20, random_state=42)

rf_model_log = RandomForestRegressor(random_state=42)
rf_model_log.fit(X_train_log, y_train_log)

# Prediction
y_pred_log = rf_model_log.predict(X_valid_log)

# Log (expm1)
y_pred_log_inverse = np.expm1(y_pred_log)

# Compare with the actual values
rmse_log = np.sqrt(mean_squared_error(y_valid, y_pred_log_inverse))
print(f"RMSE (log dönüşümlü): {rmse_log}")


# Perform hyperparameter optimization

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, None],
    "max_features": ["auto", "sqrt"]
}

grid_search = GridSearchCV(rf_model_log, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=2)
grid_search.fit(X_train_log, y_train_log)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi skor (neg RMSE): {grid_search.best_score_}")

# Examine the variable importance
importances = rf_model_log.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df["Feature"][:20][::-1], feature_importance_df["Importance"][:20][::-1])
plt.title("En Önemli 20 Özellik")
plt.xlabel("Önem Düzeyi")
plt.show()


##########################
# The dataset is part of a large dataset maintained by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) in the USA.
# The data was collected for a diabetes research study conducted on Pima Indian women aged 21 and over, living in Phoenix, the 5th largest city in Arizona, USA.
# The target variable is specified as "outcome"; 1 indicates a positive diabetes test result, while 0 indicates a negative result.
##########################


###########################
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/diabetes.csv")
#Take a look at the general overview.

def check_df(df,head=5):
 print("################### Shape ###########")
 print(df.shape)
 print("################### Types ###########")
 print(df.dtypes)
 print("################### Head ############")
 print(df.head(head))
 print("################### Tail ############")
 print(df.tail(head))
 print("################### NA ##############")
 print(df.isnull().sum())
 print("################### Quantiles #######")
 print(df.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df,head=5)

# Identify the numerical and categorical variables.
def grab_col_names(df, cat_th=18, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Analyze the numerical and categorical variables.
def num_summary(df, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[num_cols].describe(quantiles).T)

    if plot:
        df[num_cols].hist()
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show()


num_summary(df, "Glucose", plot=True)
num_summary(df, "BloodPressure", plot=True)
num_summary(df, "SkinThickness", plot=True)
num_summary(df, "Insulin", plot=True)
num_summary(df, "BMI", plot=True)
num_summary(df, "DiabetesPedigreeFunction", plot=True)
num_summary(df, "Age", plot=True)

def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("################################")

    if plot:
         sns.countplot(x=df[col_name], data=df)
         plt.show(block=True)


cat_summary(df, "Outcome", plot=True)  # Outcome kategorik bir değişken
cat_summary(df, "Pregnancies", plot=True)



# Perform target variable analysis.
# (Calculate the mean of the target variable by categorical variables,
# and calculate the mean of numerical variables by the target variable.)

def target_summary_with_cat(df, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# Fonksiyonun doğru çağrımı
target_summary_with_cat(df, "Outcome", "Pregnancies")
target_summary_with_cat(df, "Outcome", "Age")
target_summary_with_cat(df, "Outcome", "SkinThickness")
target_summary_with_cat(df, "Outcome", "Insulin")
target_summary_with_cat(df, "Outcome", "BMI")
target_summary_with_cat(df, "Outcome", "BloodPressure")
target_summary_with_cat(df, "Outcome", "Glucose")
target_summary_with_cat(df, "Outcome", "DiabetesPedigreeFunction")


## Perform outlier analysis.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Insulin")

low, up = outlier_thresholds(df, "Insulin")

df[(df["Insulin"] < low) | (df["Insulin"] > up)].head()
df[(df["Insulin"] < low) | (df["Insulin"] > up)].index

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Insulin")
check_outlier(df, "Pregnancies")
check_outlier(df, "SkinThickness")
check_outlier(df, "BMI")
check_outlier(df, "BloodPressure")
check_outlier(df, "Glucose")
check_outlier(df, "DiabetesPedigreeFunction")

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Insulin")
grab_outliers(df, "Insulin", True)

insulin_index = grab_outliers(df, "Insulin", True)
outlier_thresholds(df, "Insulin")

import seaborn as sns
import matplotlib.pyplot as plt

for col in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

for col in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Histogram - {col}")
    plt.show()


# Perform missing value analysis.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)

#  Perform correlation analysis.

import seaborn as sns
import matplotlib.pyplot as plt

# Selecting numerical variables
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

#  Calculate the correlation matrix
corr = df[num_cols].corr()

## Set the visualization size
sns.set(rc={'figure.figsize': (12, 12)})

# Create a correlation heatmap scaled between -1 and 1
sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f", vmin=-1, vmax=1, center=0)

# Show the plot
plt.show()

# "Glucose" (Blood Sugar) and "Outcome" (Diabetes) → 0.47
# There is a strong positive correlation.
# This indicates that a higher blood sugar level increases the likelihood of diabetes.
# "Insulin" and "SkinThickness" → 0.44
# There is a moderate positive relationship between insulin level and skin thickness.
# This suggests that higher insulin levels may be associated with higher skin thickness.
# "Pregnancies" and "Age" → 0.54
# As age increases, the number of pregnancies also increases.
# This is a natural relationship because the probability of pregnancy increases with age.


# Perform the necessary steps for missing and outlier values.
# There are no missing values in the dataset,
# but observations with a value of 0 in variables such as Glucose or Insulin may actually represent missing values.
# For example, a person's glucose or insulin value cannot be zero.
# Considering this, you can assign NaN to zero values in the relevant variables,
# and then apply missing value operations accordingly.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Identifying and replacing missing values
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_columns] = df[zero_columns].replace(0, np.nan)

# Fill missing values with the mean
df[zero_columns] = df[zero_columns].fillna(df[zero_columns].mean())

# Create new variables
df["Age_qcut"] = pd.qcut(df['Age'], 5, labels=False)  # Interval yerine int değerler alacak

def categorize_bmi(value):
    if value < 18.5:
        return "Underweight"
    elif 18.5 <= value < 25:
        return "Normal"
    elif 25 <= value < 30:
        return "Overweight"
    else:
        return "Obese"

df["BMI_Category"] = df["BMI"].apply(categorize_bmi)

def categorize_glucose(value):
    if value < 70:
        return "Low"
    elif 70 <= value < 140:
        return "Normal"
    else:
        return "High"

df["Glucose_Level"] = df["Glucose"].apply(categorize_glucose)


# Perform encoding operations
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
labelencoder = LabelEncoder()
for col in binary_cols:
    df[col] = labelencoder.fit_transform(df[col])

# One-Hot Encoding
df = pd.get_dummies(df, columns=["BMI_Category"], drop_first=True)

# Glucose_Level was numerically encoded using Label Encoding
df["Glucose_Level"] = labelencoder.fit_transform(df["Glucose_Level"])

# Standardize the numerical variables
num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Build the model
y = df["Outcome"].astype(int)  # 📢 Hedef değişkenin veri tipi kesinlikle `int`
X = df.drop(["Outcome"], axis=1)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

# Model prediction
y_pred = rf_model.predict(X_test)

## Evaluate the model performance
print("Accuracy Score:", accuracy_score(y_pred, y_test))
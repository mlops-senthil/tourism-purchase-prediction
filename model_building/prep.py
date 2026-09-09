
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login, HfApi
import os

# Define constants for the dataset and output paths
# api = HfApi(token=os.getenv("HF_TOKEN"))
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
DATASET_PATH = "hf://datasets/senthil31/tourism-dataset/tourism.csv"
# "hf://datasets/praneeth232/machine-failure-prediction/machine-failure-prediction.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop column 'CustomerID' if it exists as it is just an arbitrary identifier
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# -------------------------------------------------------------------
# 1. Handling Missing Values (Imputation)
# -------------------------------------------------------------------
# For numerical columns, impute with the dataset median
num_cols_to_impute = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'NumberOfTrips']
for col in num_cols_to_impute:
    df[col] = df[col].fillna(df[col].median())

# For MonthlyIncome, fill missing values using the median of their specific Designation
df['MonthlyIncome'] = df.groupby('Designation')['MonthlyIncome'].transform(lambda x: x.fillna(x.median()))

# For categorical/discrete columns, impute with the most frequent value (Mode)
cat_cols_to_impute = ['TypeofContact', 'PreferredPropertyStar', 'NumberOfChildrenVisiting']
for col in cat_cols_to_impute:
    df[col] = df[col].fillna(df[col].mode()[0])


# -------------------------------------------------------------------
# 2. Text Standardization & Categorical Data Cleaning
# -------------------------------------------------------------------
# Fix typo in Gender ('Fe Male' -> 'Female')
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Fix space in Occupation ('Free Lancer' -> 'Freelancer')
df['Occupation'] = df['Occupation'].replace('Free Lancer', 'Freelancer')


# -------------------------------------------------------------------
# 3. Outlier Treatment (Capping / Winsorization)
# -------------------------------------------------------------------
# Income Outliers: Group by designation, identify extreme errors (<10k or >3x median), replace with median
for desig in df['Designation'].unique():
    desig_mask = df['Designation'] == desig
    median_inc = df[desig_mask]['MonthlyIncome'].median()

    # Cap extreme lower error bounds (e.g. 1000.0) and severe upper bounds (e.g. 98678.0)
    df.loc[desig_mask & ((df['MonthlyIncome'] < 5000) | (df['MonthlyIncome'] > median_inc * 2.5)), 'MonthlyIncome'] = median_inc

# Cap generic numerical outliers at the 99th percentile (e.g., Pitch duration of 127 mins, Trips of 22)
outlier_features = ['DurationOfPitch', 'NumberOfTrips']
for col in outlier_features:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])


# -------------------------------------------------------------------
# 4. Categorical Encoding (Text to Numbers)
# -------------------------------------------------------------------
# A. Ordinal Encoding (Columns with an inherent order)
product_mapping = {'Basic': 1, 'Standard': 2, 'Deluxe': 3, 'Super Deluxe': 4, 'King': 5}
designation_mapping = {'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5}

df['ProductPitched'] = df['ProductPitched'].map(product_mapping)
df['Designation'] = df['Designation'].map(designation_mapping)

# B. One-Hot Encoding (Nominal columns without order)
nominal_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True) # drop_first=True avoids the dummy variable trap


# -------------------------------------------------------------------
# 5. Feature Scaling (Before balancing data)
# -------------------------------------------------------------------
# Separate Features (X) and Target (y)
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Train-Test Split (Best practice: scale data AFTER split to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit scaler on training data and transform both train and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="senthil31/tourism-dataset",
        repo_type="dataset",
    )

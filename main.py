from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def replace_missing_values(data, variables):
    print("Missing values found in the dataset.")
    missing = data.original.isnull().sum()
    print(missing)
    print("Replacing missing values with random values...")
    for col in data.original.columns[data.original.isnull().any()]:
        if variables[variables['name'] == col]['type'].values[0] == 'Integer':
            # Drop NaNs, get unique existing integers, and fill with random choices
            choices = data.original[col].dropna().unique()
            data.original[col] = data.original[col].apply(
                lambda x: np.random.choice(choices) if pd.isna(x) else x
            )
        elif variables[variables['name'] == col]['type'].values[0] == 'Categorical':
            # Same idea for categorical variables
            choices = data.original[col].dropna().unique()
            data.original[col] = data.original[col].apply(
                lambda x: np.random.choice(choices) if pd.isna(x) else x
            )
    missing_values = data.original.isnull().values.any()
    if missing_values:
        print("Still missing values after replacement.")
    else:
        print("All missing values have been replaced.")
    return data

def one_hot_encode(data, variables):
    print("One-hot encoding categorical variables...")
    categorical_cols = variables[variables['type'] == 'Categorical']['name'].values
    for col in categorical_cols:
        if col in data.original.columns:
            dummies = pd.get_dummies(data.original[col], prefix=col, drop_first=True)
            data.original = pd.concat([data.original, dummies], axis=1)
            data.original.drop(col, axis=1, inplace=True)
    return data

if __name__ == "__main__":
    heart_disease = fetch_ucirepo(id=45) 
    data = heart_disease.data
    metadata = heart_disease.metadata
    variables = heart_disease.variables

    print("Checking data for missing values...")
    missing_values = data.original.isnull().values.any()
    if missing_values:
        data = replace_missing_values(data,variables)
    else:
        print("No missing values found in the dataset.")

    print("One-hot encoding categorical variables...")
    data = one_hot_encode(data, variables)


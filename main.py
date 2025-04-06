from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score

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

    # Convert 'num' column to binary (0 or 1)
    data.original['num'] = data.original['num'].apply(lambda x: 1 if x > 0 else x)
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

def split_data(data):
    print("Splitting data into training and testing sets...")
    X = data.original.drop(columns=['num'])
    y = data.original['num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_dt(X_train, y_train):
    print("Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    return accuracy

if __name__ == "__main__":
    heart_disease = fetch_ucirepo(id=45)
    data = heart_disease.data
    metadata = heart_disease.metadata
    variables = heart_disease.variables

    print("Checking data for missing values...")
    missing_values = data.original.isnull().values.any()
    if missing_values:
        data = replace_missing_values(data, variables)
    else:
        print("No missing values found in the dataset.")

    data = one_hot_encode(data, variables)

    X_train, X_test, y_train, y_test = split_data(data)

    dt_model = train_dt(X_train, y_train)

    accuracy = evaluate_model(dt_model, X_test, y_test)

    print("Model training and evaluation completed.")




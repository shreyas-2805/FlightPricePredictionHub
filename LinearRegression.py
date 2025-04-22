# linear_model.py (refactored version of LinearRegression.py)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df

def split_data(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def save_model(model, filename="linear_regression.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

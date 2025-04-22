import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("C:/VS Code/RV Univerity/CS2004 Agile/CP3Project/Clean_Dataset.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mae")

model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

y_pred = model.predict(X_test_scaled)
print("DNN MAE:", mean_absolute_error(y_test, y_pred))

model.save("dnn_model.keras")
print("Deep Neural Network model saved as dnn_model.keras!")

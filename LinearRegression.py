import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("C:/VS Code/RV Univerity/CS2004 Agile/CP3Project/Clean_Dataset.csv")

# Drop unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split into features and target
X = df.drop(columns=["price"])
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr.predict(X_test)
print("LinearRegression MAE:", mean_absolute_error(y_test, y_pred))

# Save model
with open("linear_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
print("LinearRegression model saved as linear_regression.pkl!")

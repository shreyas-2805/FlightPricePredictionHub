import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv(r"ML Models/Clean_Dataset.csv")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0"], inplace=True)

# Initialize dictionary to hold label encoders
label_encoders = {}

# Encode categorical columns and print mappings
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Apply the label encoding
    label_encoders[col] = le  # Save the encoder for later use
    
    # Print label encoding mappings
    print(f"Label encoding mappings for {col}:")
    print(dict(zip(le.classes_, range(len(le.classes_)))))
    print()

# Features and target variable
X = df.drop(columns=["price"])
y = df["price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf.predict(X_test)
print("RandomForest MAE:", mean_absolute_error(y_test, y_pred))

# Save the trained RandomForest model
with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
print("RandomForestRegressor model saved as random_forest.pkl!")

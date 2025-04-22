import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv(r"C:/VS Code/RV Univerity/CS2004 Agile/CP3Project/Clean_Dataset.csv")

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

# Train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt.predict(X_test)
print("DecisionTree MAE:", mean_absolute_error(y_test, y_pred))

# Save model
with open("decision_tree.pkl", "wb") as f:
    pickle.dump(dt, f)
print("DecisionTreeRegressor model saved as decision_tree.pkl!")

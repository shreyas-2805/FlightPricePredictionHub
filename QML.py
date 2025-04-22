import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = r"C:\VS Code\RV Univerity\CS2004 Agile\CP3Project\Clean_Dataset.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Separate target variable (flight price)
target = "price"
X = df.drop(columns=[target])
y = df[target].values.reshape(-1, 1)

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

# Define a Quantum Device
num_qubits = X_train.shape[1]
dev = qml.device("lightning.qubit", wires=num_qubits)  # PennyLane Lightning for faster training

# Define Quantum Circuit (Variational Ansatz)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(num_qubits):
        qml.RY(inputs[i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

# Define Quantum Neural Network
class QuantumNeuralNet(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        self.q_params = nn.Parameter(torch.rand(weight_shapes["weights"]))
    
    def forward(self, x):
        return torch.stack([quantum_circuit(x_i, self.q_params) for x_i in x])

# Initialize Model
num_layers = 2
model = QuantumNeuralNet(num_qubits, num_layers)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the QNN
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save Trained Quantum Model
with open("quantum_nn.pkl", "wb") as f:
    pickle.dump(model, f)

print("Quantum Neural Network Model Saved Successfully!")

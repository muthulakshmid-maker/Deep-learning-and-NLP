import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("========== TASK 2: TRAINING LOOP ==========")


data = load_breast_cancer()

X = data.data
y = data.target

print("\nDataset Shape:", X.shape)
print("Target Shape:", y.shape)


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nTraining Data:", X_train.shape)
print("Validation Data:", X_val.shape)
print("Testing Data:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(
    X_train, dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val, dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.reshape(-1, 1), dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val.reshape(-1, 1), dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.reshape(-1, 1), dtype=torch.float32
)


class CancerMLP(nn.Module):

    def __init__(self):
        super(CancerMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


model = CancerMLP()

print("\n========== MODEL ==========")
print(model)


criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 50

train_losses = []
val_accuracies = []

print("\n========== TRAINING STARTED ==========")

for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_losses.append(loss.item())


    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val_tensor)

        val_predictions = (
            torch.sigmoid(val_outputs) >= 0.5
        ).float()

        val_accuracy = (
            val_predictions.eq(y_val_tensor)
            .float()
            .mean()
            .item()
        )

    val_accuracies.append(val_accuracy)


    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {loss.item():.4f}, "
        f"Validation Accuracy = {val_accuracy:.4f}"
    )


print("\n========== TEST EVALUATION ==========")

model.eval()

with torch.no_grad():

    test_outputs = model(X_test_tensor)

    test_predictions = (
        torch.sigmoid(test_outputs) >= 0.5
    ).float()

    pytorch_accuracy = (
        test_predictions.eq(y_test_tensor)
        .float()
        .mean()
        .item()
    )

print("PyTorch Test Accuracy:", round(pytorch_accuracy, 4))


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)

plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()

plt.savefig("task2_training_results.png")

plt.show()

print("\nTraining graphs saved successfully!")

print("\n========== LOGISTIC REGRESSION ==========")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("Logistic Regression Test Accuracy:",
      round(logistic_accuracy, 4))


print("\n========== MODEL COMPARISON ==========")

print("PyTorch Neural Network:",
      round(pytorch_accuracy, 4))

print("Scikit-learn Logistic Regression:",
      round(logistic_accuracy, 4))

print("\n========== TASK 4 COMPLETED ==========")

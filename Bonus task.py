import torch
import time

print("========== CPU vs GPU PERFORMANCE ==========")

size = 1500

# Create random matrices
A = torch.randn(size, size)
B = torch.randn(size, size)

print("\nRunning on CPU...")

start_time = time.perf_counter()

cpu_result = torch.matmul(A, B)

cpu_time = time.perf_counter() - start_time

print("CPU Result Shape:", cpu_result.shape)
print(f"CPU Time: {cpu_time:.4f} seconds")


if torch.cuda.is_available():

    print("\nRunning on GPU...")

    device = torch.device("cuda")

    A_gpu = A.to(device)
    B_gpu = B.to(device)

    torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    gpu_result = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    gpu_time = time.perf_counter() - start_time

    print("GPU Result Shape:", gpu_result.shape)
    print(f"GPU Time: {gpu_time:.4f} seconds")

    print("\n========== COMPARISON ==========")

    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.2f}x")

else:

    print("\nGPU is not available.")
    print("Only CPU performance was measured.")
    print("Use Google Colab with GPU runtime for comparison.")

print("\n========== BONUS COMPLETED ==========")











import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os

print("Current Working Directory:")
print(os.getcwd())


print("========== EARLY STOPPING ==========")

data = load_breast_cancer()

X = data.data
y = data.target


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


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.reshape(-1, 1), dtype=torch.float32
)

y_val = torch.tensor(
    y_val.reshape(-1, 1), dtype=torch.float32
)

y_test = torch.tensor(
    y_test.reshape(-1, 1), dtype=torch.float32
)


class CancerMLP(nn.Module):

    def __init__(self):
        super().__init__()

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

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 100
patience = 10

best_val_loss = float("inf")
counter = 0

best_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pth"
)


print("\n========== TRAINING STARTED ==========")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    train_loss = criterion(outputs, y_train)

    train_loss.backward()

    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)

        val_loss = criterion(val_outputs, y_val)

    current_val_loss = val_loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {train_loss.item():.4f}, "
        f"Validation Loss = {current_val_loss:.4f}"
    )

    if current_val_loss < best_val_loss:

        best_val_loss = current_val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model saved!")

    else:

        counter += 1

        print("No improvement:", counter)

    if counter >= patience:

        print("\nEarly Stopping Triggered!")

        break


print("\n========== LOADING BEST MODEL ==========")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=True
    )
)

print("Best model weights loaded successfully!")

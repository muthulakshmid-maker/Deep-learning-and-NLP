import torch

print("========== TASK 2: AUTOGRAD ==========")

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

print("\nInitial Values:")
print("w =", w.item())
print("b =", b.item())

loss = (w * 2 + b - 10) ** 2

print("\nLoss:", loss.item())


loss.backward()

print("\nCalculated Gradients:")
print("w.grad =", w.grad.item())
print("b.grad =", b.grad.item())


print("\n========== GRADIENT DESCENT ==========")

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
epochs = 30

for epoch in range(epochs):

    loss = (w * 2 + b - 10) ** 2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {loss.item():.4f}, "
            f"w = {w.item():.4f}, "
            f"b = {b.item():.4f}"
        )


print("\n========== WITHOUT ZEROING GRADIENTS ==========")

x = torch.tensor(2.0, requires_grad=True)

loss1 = x ** 2
loss1.backward()

print("After first backward:", x.grad.item())

loss2 = x ** 2
loss2.backward()

print("After second backward:", x.grad.item())

print("\n========== CONCEPTS ==========")

print("""
1. requires_grad=True tells PyTorch to track operations
   and calculate gradients for a tensor.

2. backward() calculates gradients automatically using
   backpropagation.

3. detach() creates a tensor disconnected from the
   current computation graph.
""")

detached_tensor = x.detach()

print("Detached Tensor:", detached_tensor)
print("Requires Grad:", detached_tensor.requires_grad)

print("\n========== TASK 2 COMPLETED ==========")

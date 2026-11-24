import torch
import numpy as np

print("========== PYTORCH TENSORS ==========")

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.zeros(2, 3)
tensor3 = torch.ones(2, 3)
tensor4 = torch.arange(0, 10)
tensor5 = torch.randn(3, 4)

tensors = [tensor1, tensor2, tensor3, tensor4, tensor5]

for i, tensor in enumerate(tensors, start=1):
    print(f"\nTensor {i}:")
    print(tensor)
    print("Shape:", tensor.shape)
    print("Dtype:", tensor.dtype)
    print("Device:", tensor.device)



matrix = torch.arange(1, 13).reshape(3, 4)

print("\nOriginal Matrix:")
print(matrix)

print("\nFirst Row:")
print(matrix[0, :])

print("\nFirst Column:")
print(matrix[:, 0])


view_tensor = matrix.view(2, 6)

print("\nView Reshape:")
print(view_tensor)


reshape_tensor = matrix.reshape(4, 3)

print("\nReshape:")
print(reshape_tensor)

transpose_tensor = matrix.T

print("\nTranspose:")
print(transpose_tensor)


A = torch.randn(3, 4)
B = torch.randn(4, 2)

result1 = A @ B
result2 = torch.matmul(A, B)

print("\nMatrix A Shape:", A.shape)
print("Matrix B Shape:", B.shape)

print("\nUsing @:")
print(result1)
print("Result Shape:", result1.shape)

print("\nUsing torch.matmul:")
print(result2)
print("Result Shape:", result2.shape)


numpy_array = np.array([10, 20, 30])

torch_tensor = torch.from_numpy(numpy_array)

print("\nNumPy to Tensor:")
print(torch_tensor)


converted_array = torch_tensor.numpy()

print("\nTensor to NumPy:")
print(converted_array)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nGPU Available")
else:
    device = torch.device("cpu")
    print("\nGPU Not Available. Using CPU")

gpu_tensor = torch.tensor([1, 2, 3]).to(device)

print("Final Tensor:", gpu_tensor)
print("Device:", gpu_tensor.device)

print("\n========== TASK 1 COMPLETED ==========")

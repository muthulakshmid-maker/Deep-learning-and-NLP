import torch
import torch.nn as nn

print("========== TASK 1: nn.Module ==========")


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)


model = MLP()

print("\n========== MODEL SUMMARY ==========")
print(model)


total_parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nTotal Parameters:", total_parameters)


print("\n========== EMBEDDING ==========")

embedding = nn.Embedding(10, 4)

indices = torch.tensor([1, 2, 3, 4, 5])

embedding_output = embedding(indices)

print("Input Indices:")
print(indices)

print("\nEmbedding Output:")
print(embedding_output)

print("\nEmbedding Output Shape:")
print(embedding_output.shape)


print("\n========== BATCH NORMALIZATION ==========")

batch_norm = nn.BatchNorm1d(16)

batch_data = torch.randn(8, 16)

normalized_data = batch_norm(batch_data)

print("Original Batch Shape:", batch_data.shape)

print("Normalized Batch Shape:", normalized_data.shape)

print("\nMean Per Feature:")
print(normalized_data.mean(dim=0))

print("\nStandard Deviation Per Feature:")
print(normalized_data.std(dim=0, unbiased=False))


print("\n========== TRAIN MODE ==========")

model.train()

print("Model Training Mode:", model.training)


print("\n========== EVALUATION MODE ==========")

model.eval()

print("Model Evaluation Mode:", model.training)


print("\n========== FORWARD PASS ==========")

sample_input = torch.randn(8, 16)

output = model(sample_input)

print("Input Shape:", sample_input.shape)
print("Output Shape:", output.shape)

print("\n========== TASK 3 COMPLETED ==========")

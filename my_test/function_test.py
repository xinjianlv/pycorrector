import torch
from torch import nn


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input.shape)
print(target.shape)
output = loss(input, target)
output.backward()
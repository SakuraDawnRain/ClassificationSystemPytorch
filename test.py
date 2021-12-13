import torch
import criterion

x = torch.tensor([[1,1,1]], dtype=torch.float)
y = torch.tensor([0])

c = criterion.CrossEntropyLoss(x, y)
print(c)

l = torch.nn.CrossEntropyLoss()
print(l(x,y))
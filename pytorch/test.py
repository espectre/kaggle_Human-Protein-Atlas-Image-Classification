import torch


a=torch.rand(2,3)
print(a)
print(a.max(0,keepdim=True))


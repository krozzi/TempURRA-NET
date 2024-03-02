import torch
x = torch.randn(1, 3, 128, 256)
y = torch.randn(1, 3, 128, 256)
z = torch.randn(1, 3, 128, 256)
print(x, y, z)
print(x.size(), y.size(), z.size())
a = torch.cat([x, y], dim=1)
print(a.size(), a)
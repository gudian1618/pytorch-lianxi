import torch
import numpy as np

a = torch.rand(2, 4, 1, 3)
b = torch.rand(1, 4, 3)

print(a)
print(b)
c = a + b
print(c.shape)

a = torch.rand(2, 2)
b = torch.rand(2, 2)
# a = a * 10
print(a)

print(torch.floor(a))
print(torch.ceil(a))
print(torch.round(a))
print(torch.trunc(a))
print(torch.frac(a))
print(a % 2)

print(torch.eq(a, b))
print(torch.ge(a, b))
print(torch.gt(a, b))
print(torch.le(a, b))
print(torch.lt(a, b))
print(torch.ne(a, b))

a = torch.tensor([1, 4, 4, 3, 5])
print(torch.sort(a, dim=0, descending=True))

a = torch.tensor([[2, 4, 3, 1, 5], [2, 3, 5, 1, 4]])
print(a)
print(torch.topk(a, k=1, dim=1))
print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=1))

a = torch.rand(2, 3)
# a = torch.tensor([2, 3, np.nan])
print(torch.isfinite(a))
print(torch.isfinite(a / 0))
print(torch.isinf(a / 0))
print(torch.isnan(a))

a = torch.ones(2, 3)
b = torch.cos(a)
print(a)
print(b)


import torch
import numpy as np

dev = torch.device("cpu")
dev = torch.device("cuda")
print(dev)

a = torch.Tensor([[1, 2], [3, 4]])
# a = torch.ones(2, 3)
# a = torch.randn(2, 3)
# a = torch.normal(mean=0.0, std=torch.rand(5))
# a = torch.arange(0, 11, 4)
a = torch.randperm(10)
# a = torch.linspace(0, 11, 4)
# a = torch.zeros(2, 3)

print(a)
print(a.type())

i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5])
# x = torch.sparse_coo_tensor(i, v, (4, 4)).to_dense()
x = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float, device=dev).to_dense()
print(x)

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)
print(a + b)
print(a.add(b))
print(torch.add(a, b))

print(a - b)
print(torch.sub(a, b))
print(a.sub(b))

print(a * b)
print(torch.mul(a, b))
print(a.mul(b))
# print(a.mul_(b))
print(a)

print(a / b)
print(torch.div(a, b))
print(a.div(b))

a = torch.rand(2, 3)
b = torch.rand(3, 3)

print(a @ b)
print(torch.matmul(a, b))
print(a.matmul(b))
print(a.mm(b))

x = torch.ones(1, 2, 3, 4, dtype=torch.float32)
y = torch.ones(1, 2, 4, 3, dtype=torch.float32)

print(x.matmul(y).shape)
print(x.matmul(y))
# print(x.mm(y))

a = torch.tensor([1, 2])
print(torch.pow(a, 3))
print(a.pow(3))
print(a ** 3)
print(a.pow_(3))

a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())

print(torch.log(a))
print(a.log())

print(torch.sqrt(a))
print(a.sqrt())


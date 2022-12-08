import torch

a = torch.rand(2, 2)
print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a, dim=0))
print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=0))
print(torch.std(a, dim=0))
print(torch.median(a, dim=0))
print(torch.mode(a, dim=0))

a = torch.rand(2, 2) * 10

print(torch.histc(a, 6, 0, 0))

a = torch.randint(0, 10, [10])
print(torch.bincount(a))

torch.manual_seed(2)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)
torch.normal(mean, std)
print(torch.normal(mean, std))

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a, b)
print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=0))
print(torch.dist(a, b, p=3))

print(torch.norm(a))
print(torch.norm(a, p=0))
print(torch.norm(a, p='fro'))

a = torch.rand(2, 2) * 10
print(a)
a = a.clamp(2, 5)
print(a)

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a)
print(b)
torch.where(a > 0.5, a, b)

a = torch.rand(4, 4)
print(a)
out = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print(out, out.shape)

a = torch.linspace(1, 16, 16).view(4, 4)
print(a)

out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                 [0, 1, 2, 2],
                                                 [0, 1, 3, 3]]))
print(out, out.shape)

mask = torch.gt(a, 8)
print(mask)
out = torch.masked_select(a, mask)
print(out)

b = torch.take(a, index=torch.tensor([0, 15, 13, 10]))
print(b)

a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
out = torch.nonzero(a)
print(out)

a = torch.zeros((2, 4))
b = torch.ones((2, 4))

out = torch.cat((a, b), dim=0)
out = torch.cat((a, b), dim=1)
print(out)

a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
print(a, b)
# out = torch.stack((a, b), dim=0)
out = torch.stack((a, b), dim=1)
print(out)
print(out.shape)

print(out[:, 0, :])
print(out[:, 1, :])

a = torch.rand((3, 4))
print(a)

out = torch.chunk(a, 2, dim=1)
print(out[0].shape, out[1].shape)

out = torch.split(a, 2, dim=0)
print(out)
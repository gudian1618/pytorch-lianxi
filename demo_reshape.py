import torch

a = torch.rand(2, 3)
print(a)

out = torch.reshape(a, (3, 2))
print(out)

print(torch.t(out))
print(torch.transpose(out, 0, 1))

a = torch.rand(1, 2, 3)
out = torch.transpose(a, 0, 1)
print(out.shape)

out = torch.unsqueeze(a, -1)
print(out.shape)

out = torch.unbind(a, dim=1)
print(out)
print(a)
print(torch.flip(a, dims=[1, 2]))
print(torch.rot90(a, -1, dims=[0, 2]))

a = torch.full((2, 3), 3.14)
print(a)

x = torch.full((2, 3), 3.14)
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    # print(z.to('cpu',torch.double))



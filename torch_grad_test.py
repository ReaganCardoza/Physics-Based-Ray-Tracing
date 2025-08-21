import torch

# Exponents: 2**n as floats so pow(x, exp) is smooth in x
n = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
x = torch.tensor([4., 4., 4., 4.], requires_grad=True)

exp = (2.0 ** n).to(dtype=x.dtype)   # [1., 2., 4., 8.]
y_torch = torch.pow(x, exp)
print("y_torch:", y_torch)

# Backprop wrt x
y_torch.sum().backward()
print("x.grad (torch):", x.grad)     # should be [1, 2*4, 4*4^3, 8*4^7]

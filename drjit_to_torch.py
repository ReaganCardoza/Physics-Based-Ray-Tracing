import torch 
import drjit as dr

@dr.wrap(source="torch", target="drjit")
@dr.syntax
def pow2_correct(n, x):
    for i in range(10):
        for j in range(10):
            power = dr.power(2, n) + i + j
            result = dr.power(x, power)
    return result

n = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
x = torch.tensor([4, 4, 4, 4], dtype=torch.float32, requires_grad=True)
y_correct = pow2_correct(n, x)
print("y_correct:", y_correct)
y_correct.sum().backward()
print("x.grad_correct:", x.grad)
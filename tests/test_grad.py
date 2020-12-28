import sys
import torch
import numpy as np

sys.path.append(".")

from autograd.tensor import Tensor
from autograd.value import Value


def test_single_value():
    a, b = Value(-4.0), Value(2.0)
    
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    g.backward()    
    
    assert round(g.value, 4) == 24.7041
    assert round(a.grad, 4) == 138.8338
    assert round(b.grad, 4) == 645.5773
    print("Single value: OK")


def test_backward_pass():
    x_init = np.random.randn(100, 10)
    W_init = np.random.randn(10, 32)
    m_init = np.random.randn(1, 32)
    
    def test_tensor():
        x = Tensor(x_init)
        W = Tensor(W_init)
        m = Tensor(m_init)
        
        out = (x @ W).relu()
        out = out.sigmoid()
        out = ((out * m) + m).sum()
        out.backward()
        
        return out.value, x.grad.value, W.grad.value

    def test_pytorch():
        x = torch.tensor(x_init, requires_grad=True)
        W = torch.tensor(W_init, requires_grad=True)
        m = torch.tensor(m_init)
        
        out = (x @ W).relu()
        out = torch.sigmoid(out)
        out = ((out * m) + m).sum()
        out.backward()
        
        return out.detach().numpy(), x.grad, W.grad
    
    for x, y in zip(test_tensor(), test_pytorch()):
        assert np.allclose(x, y, atol=1e-5)
    print("Backward pass: OK")


if __name__ == "__main__":
    test_single_value()
    test_backward_pass()
import sys
import torch
import numpy as np

sys.path.append(".")

from autograd.tensor import Tensor


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
    test_backward_pass()
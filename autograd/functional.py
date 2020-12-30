import numpy as np

from scipy.special import expit

try:
    import tensor
except ModuleNotFoundError:
    from . import tensor


def check_input(value):
    if isinstance(value, (int, float)):
        return tensor.Tensor([value])
    elif isinstance(value, (list, np.ndarray)):
        return tensor.Tensor(value)
    return value


class Context:
    def __init__(self):
        self.saved = []
        
    def save_for_backward(self, *args):
        self.saved.extend(args)


class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs):
        ctx = Context()
    
        args = [check_input(arg) for arg in args]
        out = self.forward(ctx, *args, **kwargs)
        
        assert isinstance(out, tensor.Tensor), "function should return autograd.Tensor"
        
        out.children = [child for child in args if isinstance(child, tensor.Tensor)]
        out._backward = lambda grad_in: self.backward(ctx, grad_in)
            
        return out
    
    
class Add(Function):
    @staticmethod
    def forward(ctx, x, y) :
        return tensor.Tensor(x.value + y.value)
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in, grad_in]


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return tensor.Tensor(x.value * y.value)
    
    @staticmethod
    def backward(ctx, grad_in):
        x, y = ctx.saved
        return [grad_in * y.value, grad_in * x.value]


class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return tensor.Tensor(x.value @ y.value)
        
    @staticmethod
    def backward(ctx, grad_in):
        x, y = ctx.saved        
        return [grad_in @ y.value.T, x.value.T @ grad_in]


class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return tensor.Tensor(x.value ** y.value)
    
    @staticmethod
    def backward(ctx, grad_in):
        x, y = ctx.saved
        
        dx = y.value * x.value ** (y.value - 1)
        # TODO: may cause inf if base x < 0
        dy = x.value ** y.value * np.log(x.value)
        
        return [grad_in * dx, grad_in * dy] 


class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None):
        # TODO: add axis for summation
        return tensor.Tensor(x.value.sum())
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in]


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):    
        ctx.exp = expit(x.value)
        return tensor.Tensor(ctx.exp)
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in * ctx.exp * (1 - ctx.exp)]
    

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return tensor.Tensor(np.maximum(0, x.value))
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in * (ctx.x.value >= 0)]
    

class Norm(Function):
    @staticmethod
    def forward(ctx, x):  # frobenius norm (even for matrices)
        norm = np.linalg.norm(x.value)
        ctx.save_for_backward(x, norm) 
        
        return tensor.Tensor(norm)
    
    @staticmethod
    def backward(ctx, grad_in):
        x, norm = ctx.saved
        return [grad_in * (x.value / norm)]


class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return tensor.Tensor(np.log(x.value))
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in / ctx.x.value]


add, mul, matmul, pow = Add(), Mul(), MatMul(), Pow()
sum, norm, sigmoid, relu = Sum(), Norm(), Sigmoid(), ReLU()
log = Log()


def softmax(x, axis=None):
    exp = pow(np.e, x)
    return exp / sum(exp, axis=axis)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    test = tensor.Tensor.uniform(-10, 10, shape=(10, 1))
    test_t = torch.tensor(test.value, requires_grad=True)
    

    z1 = F.softmax(test_t, dim=0)
    z2 = softmax(test)
    
    z1.backward(torch.ones_like(z1))
    z2.backward()

    print(test_t.grad)
    print(test.grad)
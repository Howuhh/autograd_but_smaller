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
        ctx, args = Context(), [check_input(arg) for arg in args]
    
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


# loses preicison with x * y**(-1), so a distinct implementation for div
class Div(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return tensor.Tensor(x.value / y.value)
    
    @staticmethod
    def backward(ctx, grad_in):
        x, y = ctx.saved
        return [grad_in * (1 / y.value), grad_in * -(x.value / y.value**2)]


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
        ctx.save_for_backward(x, axis)
        return tensor.Tensor(x.value.sum(axis=axis, keepdims=True))
    
    @staticmethod
    def backward(ctx, grad_in):
        return [grad_in]   # maybe this does not work when sum over multiple axes, need some reshaping work
    

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


add, mul, div, pow, log = Add(), Mul(), Div(), Pow(), Log()
matmul, sum, norm, sigmoid, relu = MatMul(), Sum(), Norm(), Sigmoid(), ReLU()


def softmax(x, axis=None):
    exp = np.e ** x
    return exp / exp.sum(axis=axis)

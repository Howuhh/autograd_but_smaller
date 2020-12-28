import numpy as np

from .tensor import Tensor


class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def parameters(self):
        params = []
        
        def _parameters(node):
            if isinstance(node, Tensor):
                params.append(node)
            elif hasattr(node, "__dict__"):
                for _, v in node.__dict__.items():
                    _parameters(v)
        _parameters(self)
        
        return params
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class Linear(Module):
    def __init__(self, in_shape, out_shape):     
        scale = 1 / np.sqrt(in_shape)
        
        self.W = Tensor.uniform(-scale, scale, (in_shape, out_shape))
        self.b = Tensor.zeros((1, out_shape))
        
    def forward(self, X):
        return (X @ self.W) + self.b
import numpy as np

from scipy.special import expit

try:
    from .tensor import Tensor
except ModuleNotFoundError:
    from tensor import Tensor


def check_input(value):
    if isinstance(value, (int, float)):
        return Tensor([value])
    elif isinstance(value, (list, np.ndarray)):
        return Tensor(value)
    return value


class Function:
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        args = [check_input(arg) for arg in args]
    
        out = self.forward(*args, **kwargs)
        
        assert isinstance(out, Tensor), "function should return autograd.Tensor"
        
        out.children = [child for child in args if isinstance(child, Tensor)]
        out._backward = self.backward
            
        return out
    
    
class Add(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.value + y.value)
    
    def backward(self, grad_in):
        return [grad_in, grad_in]


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x, self.y = x, y
        
        return Tensor(x.value * y.value)
    
    def backward(self, grad_in):
        return [grad_in * self.y.value, grad_in * self.x.value]


class MatMul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        
        return Tensor(x.value @ y.value)
        
    def backward(self, grad_in):
        # Y = XW
        # dY/dX = dD @ W.t
        # dY/dW = X.T @ dD
        return [grad_in @ self.y.value.T, self.x.value.T @ grad_in]

class Pow(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        
        return Tensor(x.value ** y.value)  # only for Tensor^(int/float) for now
    
    def backward(self, grad_in):
        dx = self.y.value * self.x.value ** (self.y.value - 1)
        dy = self.x.value ** self.y.value * np.log(self.x.value)
        
        return [grad_in * dx, grad_in * dy] 


class Sum(Function):
    def forward(self, x):
        return Tensor(x.value.sum())
    
    def backward(self, grad_in):
        return [grad_in]


class Sigmoid(Function):
    def forward(self, x):
        self.exp = expit(x.value)
        return Tensor(self.exp)
    
    def backward(self, grad_in):
        return [grad_in * self.exp * (1 - self.exp)]
    

class ReLU(Function):
    def forward(self, x):
        self.x = x
        return Tensor(np.maximum(0, x.value))
    
    def backward(self, grad_in):
        return [grad_in * (self.x >= 0)]


add = Add()    
mul = Mul()
pow = Pow()
sum = Sum()
sigmoid = Sigmoid()
relu = ReLU()

    
if __name__ == "__main__":    
    test1 = Tensor(np.ones(5) * 5).reshape(-1, 1)
    test2 = Tensor(np.ones(5) * 2).reshape(-1, 1)
    
    z1 = test1 * test2
    z2 = mul(test1, test2)
    
    
    print(z1)
    z1.backward()
    
    print(test1.grad, test2.grad)
    test1.grad = None
    test2.grad = None
    
    print(z2)
    z2.backward()
    print(test1.grad, test2.grad)
    
    
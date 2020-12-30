import numpy as np

try:
    import functional as F
except ModuleNotFoundError:
    from . import functional as F

from scipy.special import expit
from collections import deque


class Tensor:
    def __init__(self, value):
        if isinstance(value, (int, float)):
            value = [value]

        self.value = np.array(value)
        
        self.grad = None
        self.children = []
        
    @property
    def leaf_node(self):
        return not bool(self.children)
        
    @property
    def shape(self):
        return self.value.shape
    
    @property
    def safe_grad(self):
        return self.zeros(self.shape) if self.grad is None else self.grad
    
    # magick, не до коцна понял когда нужно суммировать (?)
    @staticmethod
    def unbroadcast(out, in_shape):
        sum_axis = None
        # sum all axis with in_shape[i] < grad.shape[i]
        if in_shape != (1,):
            sum_axis = tuple([i for i in range(len(in_shape)) if in_shape[i] == 1 and out.shape[i] > 1])

        return Tensor(out.value.sum(axis=sum_axis).reshape(in_shape))
    
    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))
    
    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))
        
    @staticmethod
    def uniform(low=0.0, high=1.0, shape=None):
        return Tensor(np.random.uniform(low, high, size=shape))
        
    @staticmethod
    def topsort(root):
        sort, visited = deque(), set()
        
        def dfs(node):
            for child in node.children:
                if child not in visited:
                    visited.add(child)
                    dfs(child)
            sort.appendleft(node)
        
        dfs(root)
        return sort
    
    def backward(self):
        topsort = self.topsort(self)
        self.grad = self.ones(self.shape)
        
        for root in topsort:
            if not root.leaf_node:
                out_grad = root._backward(root.grad.value)
                
                for i, child in enumerate(root.children):  
                    child.grad = child.safe_grad + out_grad[i]
                    child.grad = self.unbroadcast(child.grad, child.shape)
                    
    def reshape(self, *shapes):
        return Tensor(self.value.reshape(*shapes))
    
    def sum(self):
        return F.sum(self)
    
    def norm(self):
        return F.norm(self)
    
    def sigmoid(self):
        return F.sigmoid(self)
    
    def relu(self):
        return F.relu(self)
    
    def __matmul__(self, other):
        return F.matmul(self, other)
    
    def __rmatmul__(self, other):
        return F.matmul(other, self)

    def __add__(self, other):
        return F.add(self, other)
    
    def __radd__(self, other):
        return F.add(other, self)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __iadd__(self, other):
        other = F.check_input(other)
        self.value = self.value + other.value
        
        return self
    
    def __isub__(self, other):
        other = F.check_input(other)
        self.value = self.value - other.value
        
        return self
    
    def __mul__(self, other):
        return F.mul(self, other)

    def __rmul__(self, other):
        return F.mul(other, self)
    
    def __pow__(self, other):
        return F.pow(self, other)
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __repr__(self):
        array_repr = ",\n".join([7*" " + str(line) if i > 0 else str(line) for i, line in enumerate(self.value)])
                
        return f"Tensor({array_repr})"
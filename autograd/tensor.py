import numpy as np

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
    
    @staticmethod
    def check_input(value):
        if isinstance(value, (int, float)):
            return Tensor([value])
        elif isinstance(value, (list, np.ndarray)):
            return Tensor(value)
        return value
    
    # magick, не до коцна понял когда нужно суммировать (?)
    @staticmethod
    def unbroadcast(out, in_shape):
        sum_axis = None
        # Need to sum all axis with 1 = in_shape[i] < out.shape[i]
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
    def sum(tensor):
        return tensor.sum()
        
    @staticmethod
    def norm(tensor):
        return tensor.norm()
        
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
        node = Tensor(self.value.sum(axis=None))
        
        def _backward(din):
            return [din]
        
        node._backward = _backward
        node.children = [self]
        
        return node
    
    # TODO: this also need backward!
    def norm(self):
        return np.linalg.norm(self.value)
    
    def sigmoid(self):
        exp = expit(self.value)
        
        node = Tensor(exp)
        
        def _backward(din):
            return [din * exp * (1 - exp)]
        
        node._backward = _backward
        node.children = [self]
        
        return node
    
    def relu(self):
        node = Tensor(np.maximum(0, self.value))
        
        def _backward(din):
            return [din * (self.value >= 0)]
        
        node._backward = _backward
        node.children = [self]
        
        return node
    
    def __matmul__(self, other):
        # http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        node = Tensor(self.value @ other.value)
        
        def _backward(din):
            return [din @ other.value.T, self.value.T @ din]
        
        node._backward = _backward
        node.children = [self, other]
        
        return node
    
    def __rmatmul__(self, other):
        pass

    def __add__(self, other):
        other = self.check_input(other)
        
        node = Tensor(self.value + other.value)
        
        def _backward(din):
            return [din, din]

        node._backward = _backward
        node.children = [self, other]
        
        return node
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __iadd__(self, other):
        other = self.check_input(other)
        self.value = self.value + other.value
        
        return self
    
    def __isub__(self, other):
        other = self.check_input(other)
        self.value = self.value - other.value
        
        return self
    
    def __mul__(self, other):
        other = self.check_input(other)
        
        node = Tensor(self.value * other.value)
        
        def _backward(din):
            return [din * other.value, din * self.value]

        node._backward = _backward
        node.children = [self, other]
        
        return node

    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        
        node = Tensor(self.value ** other)
        
        def _backward(din):
            return [din * (other * self.value ** (other - 1))]
        
        node._backward = _backward
        node.children = [self]
        
        return node
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * (1 / other)
    
    def __repr__(self):
        array_repr = ",\n".join([7*" " + str(line) if i > 0 else str(line) for i, line in enumerate(self.value)])
                
        return f"Tensor({array_repr})"
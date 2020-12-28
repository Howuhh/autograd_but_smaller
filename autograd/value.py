import math

from collections import deque

class Value:
    def __init__(self, value):
        self.value = value
        
        self.grad = None
        self.local_grad = None
        
        self.children = []
        
    @property
    def safe_grad(self):
        return 0.0 if self.grad is None else self.grad
    
    # TODO: this is bottleneck for python: recursion
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
    
    @staticmethod
    def check_input(value):
        return value if isinstance(value, Value) else Value(value)
    
    def backward(self):
        topsort = self.topsort(self)
        topsort[0].grad = 1.0
        
        for root in topsort:
            for i, child in enumerate(root.children):
                child.grad = child.safe_grad + (root.local_grad[i] * root.grad)        

    def relu(self):
        node = Value(max(0, self.value))
                
        node.children = [self]
        node.local_grad = [0 if self.value < 0 else 1]
        
        return node
                
    def __add__(self, other):
        other = self.check_input(other)
        
        node = Value(self.value + other.value)
        
        node.children = [self, other]
        node.local_grad = [1, 1]
         
        return node
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = self.check_input(other)
        
        node = Value(self.value * other.value)
        
        node.children = [self, other]
        node.local_grad = [other.value, self.value]
    
        return node
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (float, int))
        
        node = Value(self.value ** other)

        node.children = [self]
        node.local_grad = [other * self.value ** (other - 1)]
        
        return node

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        return other * self**(-1)

    def __repr__(self):
        return f"Value({self.value})"
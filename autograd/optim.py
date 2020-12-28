
class SGD:
    def __init__(self, parameters, lr=1e-3, normalize=False):
        self.parameters = parameters
        self.lr = lr
        self.normalize = normalize
        
    def step(self):
        for param in self.parameters:
            if self.normalize:
                step = param.grad / param.grad.norm()
            else:
                step = param.grad
                
            param -= self.lr * step
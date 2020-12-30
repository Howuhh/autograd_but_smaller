import sys
import numpy as np

sys.path.append(".")

import autograd.nn as nn
import autograd.optim as optim

from autograd.validation import MSELoss
from autograd.tensor import Tensor
from sklearn.datasets import load_boston


class SimpleRegNet(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        
        self.l1 = nn.Linear(in_shape, 32)
        self.l2 = nn.Linear(32, 1)
                    
    def __call__(self, X):
        out = self.l1(X).relu()
        return self.l2(out)
    

def test_boston():
    from tqdm import tqdm
    
    X_, y_ = load_boston()["data"], load_boston()["target"]
    X, y = Tensor(X_), Tensor(y_).reshape(-1, 1)
    
    net = SimpleRegNet(X.shape[1])
    optimizer = optim.SGD(net.parameters(), lr=1e-5)

    for i in tqdm(range(200)):
        net.zero_grad()
        
        loss = MSELoss(net(X), y)
        loss.backward()
        optimizer.step()
    
    print("RMSE:", np.sqrt(loss.value[0]))
    assert np.sqrt(loss.value[0]) <= 20


if __name__ == "__main__":
    test_boston()
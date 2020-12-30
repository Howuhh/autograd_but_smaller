import numpy as np

try:
    import functional as F
    from tensor import Tensor
except ModuleNotFoundError:
    from . import functional as F
    from .tensor import Tensor
    

def one_hot_encode(labels):
    ohe = np.zeros((labels.size, labels.max() + 1))
    ohe[np.arange(labels.size), labels] = 1
    
    return Tensor(ohe)


def MSELoss(y_pred, y_true):    
    return F.sum((y_pred - y_true)**2) / y_true.shape[0] 


def CrossEntropyLoss(y_pred, y_true):    
    return -F.sum(y_true * F.log(y_pred)) / y_true.shape[0]


if __name__ == "__main__":
    import torch
    
    test = Tensor.uniform(0, 10, shape=(5, 3))
    labels = Tensor([0, 1, 0, 2, 3])
    
    labels_ohe = one_hot_encode(labels.value)
    print(labels_ohe)
    print(test)
    
    # CrossEntropyLoss()
    
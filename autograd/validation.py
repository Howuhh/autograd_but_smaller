from .tensor import Tensor


def MSELoss(y_pred, y_true):    
    return Tensor.sum((y_pred - y_true)**2) / y_true.shape[0] 

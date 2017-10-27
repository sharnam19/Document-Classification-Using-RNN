import numpy as np
from scipy import signal

def affine_forward(x, w, b):
    """
        Input
        x: Input of shape (N,D)
        w: Weights of shape (D,H)
        b: Biases of shape (H,)

        Output(out,cache)
        Returns out of shape (N,H)

    """
    out = x.dot(w)+b
    cache = (x,w,b)
    return out,cache

def affine_backward(dOut,cache):
    """
        Input:
        dOut: Upstream gradient of shape (N,H)
        Returns Derivative wrt the inputs x,w,b i.e
        dx,dw,db
    """
    x,w,b = cache
    dx = np.dot(dOut,w.T)
    dw = np.dot(x.T,dOut)
    db = np.sum(dOut,axis=0)
        
    return dx,dw,db
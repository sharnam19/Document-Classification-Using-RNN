import numpy as np
def softmax_loss(x,y=None):
    """
        Input:
            x of shape (N,D)
            y of shape (N,). It is the class values
        Output:
            loss: Loss Value
            dx: Softmax Loss wrt the input. Same Shape as x
    """
    maximum = np.max(x,axis=1)
    shifted_x = x - maximum[:,np.newaxis]
    exp_x = np.exp(shifted_x)
    scores = exp_x/np.sum(exp_x,axis=1)[:,np.newaxis]
    N,D = x.shape
    predicted = np.argmax(scores,axis=1)
    if y is None:
        return predicted
    loss = 0
    loss += np.sum(-np.log(scores[range(N),y]))/N
    
    offset = np.zeros_like(scores)
    offset[range(N),y]=1
    dx = (scores-offset)/N
    return predicted,loss,dx
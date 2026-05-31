import numpy as np

def predict(X,w,b):
    return X@w+b


def gradients(X,y,y_hat,n):
    grad_w = 2/n*(X.T)@(y_hat-y)
    grad_b = 2/n*np.sum(y_hat-y)
    return grad_w,grad_b

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X)
    y = np.array(y)
    n,d = X.shape
    w = np.zeros(d)
    b = 0
    for epoch in range(epochs):
        y_hat = predict(X,w,b)
        grad_w,grad_b = gradients(X,y,y_hat,n)
        w -= grad_w*lr
        b -= grad_b*lr
    return w,b
        
    

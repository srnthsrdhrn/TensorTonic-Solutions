import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def pred(X,w,b):
    z = X@w+b
    return sigmoid(z)

def gradient(X,y,y_hat,n):
    grad_w = 1/n * (X.T@(y_hat-y))
    grad_b = 1/n * np.sum(y_hat-y)
    return grad_w,grad_b

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X)
    y = np.array(y)
    n,d = X.shape
    w = np.zeros(d)
    b = 0
    for epoch in range(n_iters):
        y_hat = pred(X,w,b)
        grad_w,grad_b = gradient(X,y,y_hat,n)
        w -= grad_w*lr
        b -= grad_b*lr
    return w,b

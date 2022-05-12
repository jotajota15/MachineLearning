
from lib2to3.pytree import LeafPattern
from operator import le
import numpy as np
import sys
import itertools
def MSE(y_true, y_predict):
    return (y_true - y_predict).pow(2).mean()

def MAE(y_true, y_predict): 
    return ((y_true - y_predict)).abs().mean()

# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - ∑(f(xj))/n )2
# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - f(x)mean )2
def score(y_true, y_predict):
    SSR = (y_true - y_predict).pow(2) # Sum of squared regression
    SST = (y_true - y_predict.mean()).pow(2) # Sum of squared total 
    return SSR / SST
REGULARIZATION = ["l1", "lasso","l2" "ridge", "elastic-net",None]

class LR:
    def __init__(self):
        return
    
    def GDMAE(x,y_true,y_predict):
        return (np.sign(y_true-y_predict).mul(x)).mean()
    
    def GDMSE(x,y_true,y_predict):
        return (2*(y_true-y_predict)).mul(x).mean()


    def fit(self,x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lambdaV=0):
        currentError = 0
        maxError = sys.float_info.max
        C = np.random.rand(x.shape[1],1) 
        prevDC = 0
        for _ in itertools.repeat(None, max_epochs):
            y_predict = x*C
            currentError = MSE(y,y_predict)
            if maxError - currentError> threshold:
                break
            maxError = currentError
            dC = self.GDMSE(x,y_predict,y) 
            C -= learning_rate * (dC+ momentum * prevDC)
            learning_rate = learning_rate/(1+decay)
            prevDC = dC
            # TODO 3: Como pongo la regularizacion, va en el error todo bien, pero en las derivadas?

        self.C = C

    def predict(self, x):
        return x * self.C
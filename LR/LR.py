
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
        C = np.random.rand(5,1) # TODO 1 : Ask for better options

    def fit(self, x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lambdaV=0):
        funcError = MSE
        if error == 'mae':
            funcError = MAE
        
    def gradientDescend(x,y_true,y_predict):
        return (2*(y_true-y_predict)).mul(x).mean()

    def fit(self,x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lambdaV=0):
        currentError = 0
        error = sys. float_info. max
        C = np.random.rand(x.shape[1],1) # TODO 1 : Ask for better options, to crate random values
        for _ in itertools.repeat(None, max_epochs):
            y_predict = x*C
            currentError = MAE(y,y_predict)
            if currentError - error > threshold:
                break
            currentError = error
            dC = self.gradientDescend(x,y_predict,y)
            C = learning_rate * dC
        self.C = C

    def predict(self, x):
        return x * self.C
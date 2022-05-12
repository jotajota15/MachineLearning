
from lib2to3.pytree import LeafPattern
from operator import le
import numpy as np
import sys
import itertools
def MSE(y_true, y_predict):
    return (y_true - y_predict).pow(2).mean()

def MAE(y_true, y_predict): 
    return ((y_true - y_predict)).abs().mean()

def GDMAE(x,y_true,y_predict):
    return (np.sign(y_true-y_predict).mul(x)).mean()
    
def GDMSE(x,y_true,y_predict):
    return (2*(y_true-y_predict)).mul(x).mean()

# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - ∑(f(xj))/n )2
# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - f(x)mean )2
def score(y_true, y_predict):
    SSR = (y_true - y_predict).pow(2) # Sum of squared regression
    SST = (y_true - y_predict.mean()).pow(2) # Sum of squared total 
    return SSR / SST
REGULARIZATION = ["l1", "lasso","l2" "ridge", "elastic-net",None]

def L1():
    pass
def L2():
    pass
def ELASTIC():
    return L2() + L1()

def L1D():
    pass
def L2D():
    pass
def ELASTICD():
    return L2D() + L1D()

class LR:
    def __init__(self):
        return


    def fit(self,x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lambdaV=0):
        # 1. Se pone los primeros atributos
        currentError = 0 # Para llevar el error
        maxError = sys.float_info.max # Para comparar en el threshold
        C = np.random.rand(x.shape[1],1)  # Para obtener los pesos iniciales de C
        prevDC = 0 # Para poder utilizar el valor previo de las derivada de C al obtener el momentum
        # 2. Se asignan las funciones de error y de la obtencion de los gradiantes
        errorFun = MAE if error == "mae" else MSE # Se elige la funcion de error
        errorDer = GDMAE if error == "mae" else GDMSE # Se elige la funcion para obtener los valores de dC
        # 3. Se pone las funciones de regularizacion con sus efectos en los gradiantes
        if regularization == None: # Si no existe una regularizacion se utiliza funciones lambda que retornan 0
            errorReg = lambda x,y: 0 
            regDer =  lambda x,y: 0 
        else: # Si si existe regularizacion se asigna las funciones a ser utilizadas para cada uno de los casos respectivos
            errorReg = ELASTIC if regularization == "elastic-net" else L1 if regularization =="l1" or regularization == "lasso" else L2
            regDer = ELASTICD if regularization == "elastic-net" else L1D if regularization =="l1" or regularization == "lasso" else L2D
        # 4. Se inicia el fit como tal
        for _ in itertools.repeat(None, max_epochs): # Se repite por n epocas
            y_predict = x*C # Se realiza la prediccion
            currentError= errorFun(y,y_predict) + errorReg()# Se calcula el error
            if maxError - currentError> threshold: # Si no se supera el threshold, se finaliza el proceso del fit
                break
            maxError = currentError # Se asigna el error nuevo maximo que se tiene
            dC = errorDer(x,y_predict,y) + regDer() # Se obtiene los valores de dC
            C -= learning_rate * (dC+ momentum * prevDC) # Se calculan los nuevos C
            learning_rate = learning_rate/(1+decay) # Se calcula la nueva taza de aprendizaje, si hay decaimiento
            prevDC = dC # Se asigna los actuales dC como los antiguos, con objetivo de que se puedan utilizar posteriormente cuando se realice el momentum
        # 5. Se asigna los pesos C
        self.C = C

    def predict(self, x):
        return x * self.C
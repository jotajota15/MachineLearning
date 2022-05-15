
from lib2to3.pytree import LeafPattern
from operator import le
import numpy as np
import sys
import itertools
def MSE(y_true, y_predict):
    return np.mean(np.power((y_true - y_predict),2))

def MAE(y_true, y_predict): 
    return np.mean(np.absolute(y_true - y_predict))

def GDMAE(x,y_predict,y_true):
    return (np.sign(y_predict-y_true).dot(x))/y_true.size
    
def GDMSE(x,y_predict,y_true):
    return ((2*(y_predict-y_true)).dot(x))/y_true.size

# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - ∑(f(xj))/n )2
# R2 = ∑ (yi - f(xi))2 /  ∑ (yi - f(x)mean )2
def score(y_true, y_predict):
    SSR = (y_true - y_predict).pow(2).sum()# Sum of squared regression
    SST = (y_true - y_predict.mean()).pow(2).sum() # Sum of squared total 
    return SSR / SST

def L1(C,lamda):
    return np.sum(np.absolute(C))*lamda

def L2(C,lamda):
    return np.power(C,2).sum()*lamda 
def ELASTIC(C,lamda):
    return L2(C,lamda) + L1(C,(1-lamda))

def L1D(C,lamda):
    return np.full(C.size,lamda)
def L2D(C,lamda):
    return C * 2 * lamda
def ELASTICD(C,lamda):
    return L2D(C,lamda) + L1D(C,(1-lamda))

class LR:
    def __init__(self):
        return

    def fit(self,x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lamda=0):
        # 1. Se pone los primeros atributos
        currentError = 0 # Para llevar el error
        first = False
        maxError = sys.maxsize # Para comparar en el threshold
        C = np.random.random(x.shape[1]+1)  # Para obtener los pesos iniciales de C
        prevDC = 0 # Para poder utilizar el valor previo de las derivada de C al obtener el momentum
        # 2. Se asignan las funciones de error y de la obtencion de los gradiantes
        errorFun = MAE if error == "mae" else MSE # Se elige la funcion de error
        errorDer = GDMAE if error == "mae" else GDMSE # Se elige la funcion para obtener los valores de dC
        # 3. Se pone las funciones de regularizacion con sus efectos en los gradiantes
        if regularization == 'none': # Si no existe una regularizacion se utiliza funciones lambda que retornan 0
            errorReg = lambda x,y: 0 
            regDer =  lambda x,y: 0 
        else: # Si si existe regularizacion se asigna las funciones a ser utilizadas para cada uno de los casos respectivos
            errorReg = ELASTIC if regularization == "elastic-net" else L1 if regularization =="l1" or regularization == "lasso" else L2
            regDer = ELASTICD if regularization == "elastic-net" else L1D if regularization =="l1" or regularization == "lasso" else L2D
        # 4. Se inicia el fit como tal
        x = np.insert(x.to_numpy(), 0, np.full(x.shape[0],1), axis=1)
        y = y.to_numpy()

        for _ in itertools.repeat(None, max_epochs): # Se repite por n epocas
            y_predict = x.dot(C) # Se realiza la prediccion
            currentError= errorFun(y,y_predict) + errorReg(C,lamda)# Se calcula el error
            if (first and  1-(currentError/maxError) > (threshold)): # Si no se supera el threshold, se finaliza el proceso del fit
                break
            first = True
            maxError = currentError # Se asigna el error nuevo maximo que se tiene
            dC = errorDer(x,y_predict,y) + regDer(C,lamda) # Se obtiene los valores de dC
            C -= learning_rate * (dC+ momentum * prevDC) # Se calculan los nuevos C
            learning_rate = learning_rate/(1+decay) # Se calcula la nueva taza de aprendizaje, si hay decaimiento
            prevDC = dC # Se asigna los actuales dC como los antiguos, con objetivo de que se puedan utilizar posteriormente cuando se realice el momentum
        # 5. Se asigna los pesos C
        self.C = C

    def predict(self, x):
        x = np.insert(x.to_numpy(), 0, np.full(x.shape[0],1), axis=1)
        return x.dot(self.C)
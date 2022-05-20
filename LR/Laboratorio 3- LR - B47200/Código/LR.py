
from lib2to3.pytree import LeafPattern
from operator import le
import numpy as np
import sys
import itertools
def MSE(y_true, y_predict):
    '''
    Funcion del error medio cuadratico
    '''
    return np.mean(np.power((y_true - y_predict),2))

def MAE(y_true, y_predict): 
    '''
    Función del error medio absoluto
    '''
    return np.mean(np.absolute(y_true - y_predict))

def GDMAE(x,y_predict,y_true):
    '''
    Función de la derivada del del error medio cuadrado
    '''
    return (np.sign(y_predict-y_true).dot(x))/y_true.size
    
def GDMSE(x,y_predict,y_true):
    '''
    Función de la derivada del del error medio absoluto
    '''
    return ((2*(y_predict-y_true)).dot(x))/y_true.size

def L1(C,lamda):
    '''
    Funcion de L1
    '''
    return np.sum(np.absolute(C))*lamda

def L2(C,lamda):
    '''
    Funcion de L2
    '''
    return np.power(C,2).sum()*lamda 
def ELASTIC(C,lamda):
    '''
    Funcion de elastic-net
    '''
    return L2(C,lamda) + L1(C,(1-lamda))

def L1D(C,lamda):
    '''
    Funcion de la derivada de L1
    '''
    return np.full(C.size,lamda)
def L2D(C,lamda):
    '''
    Funcion de la derivada de L2
    '''
    return C * 2 * lamda
def ELASTICD(C,lamda):
    '''
    Funcion de la derivada de elastic-net
    '''
    return L2D(C,lamda) + L1D(C,(1-lamda))

ERRORFUNCTION = {'mse': MSE, 'mae':MAE }
DERIVATIVEERROR = {'mse': GDMSE, 'mae':GDMAE }
REGULARIZATION = {"elastic-net" : ELASTIC,"l1":L1,"lasso":L1,"ridge":L2,"l2":L2,"none":lambda x,y: 0}
DERREGULARIZATION = {"elastic-net" : ELASTICD,"l1":L1D,"lasso":L1D,"ridge":L2D,"l2":L2D,"none":lambda x,y: 0}


class LR:
    def __init__(self):
        return

    def fit(self,x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error = 'mse', regularization='none', lamda=0):
        '''
        Funcion para obtener los valores de C o pesos para la regresion lineal
        '''
        # 1. Se pone los primeros atributos
        currentError = 0 # Para llevar el error
        first = False
        maxError = sys.maxsize # Para comparar en el threshold
        C = np.random.random(x.shape[1]+1)  # Para obtener los pesos iniciales de C
        prevDC = 0 # Para poder utilizar el valor previo de las derivada de C al obtener el momentum
        # 2. Se asignan las funciones de error y de la obtencion de los gradiantes
        errorFun = ERRORFUNCTION[error] # Se elige la funcion de error
        errorDer = DERIVATIVEERROR[error]  # Se elige la funcion para obtener los valores de dC
        # 3. Se pone las funciones de regularizacion con sus efectos en los gradiantes, si no existe una regularizacion se utiliza funciones lambda que retornan 0
        errorReg = REGULARIZATION[regularization]
        regDer = DERREGULARIZATION[regularization]
        # 4. Se inicia el fit como tal
        x = np.insert(x.to_numpy(), 0, np.full(x.shape[0],1), axis=1) # Se obtiene los valores de x como numpy y ademas se agrega el bias
        y = y.to_numpy() # Se convierte el valor de y como numpy

        for _ in itertools.repeat(None, max_epochs): # Se repite por n epocas
            y_predict = x.dot(C) # Se realiza la prediccion (Multiplicacion de matriz)
            currentError= errorFun(y,y_predict) + errorReg(C,lamda)# Se calcula el error
            if (first and  np.absolute(1-(currentError/maxError))< (threshold)): # Si no se supera el threshold, se finaliza el proceso del fit
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
        '''
        Funcion de prediccion
        '''
        x = np.insert(x.to_numpy(), 0, np.full(x.shape[0],1), axis=1) # Se obtiene los valores de x como numpy y ademas se agrega el bias
        return x.dot(self.C)

import numpy as np

class myPCA:

    def __init__(self,matrixArray):
        self.myMatrix = self.convertMatrix(matrixArray)
        self.correlationMatrix = self.correlation()
        

        self.eigVal, self.eigVec = np.linalg.eigh(self.correlationMatrix)
        self.eigVal = np.absolute(self.eigVal)
        sortIndex = np.argsort(self.eigVal)
        sortIndex

    def convertMatrix(self,matrixArray):
        '''
        Centrar y convertir matriz
        '''
        mean = [np.mean(value) for value in matrixArray]
        desviacion = [np.std(value) for value in matrixArray]


        for iy, ix in np.ndindex(matrixArray.shape):
            matrixArray[iy, ix] = (matrixArray[iy, ix] - mean[iy]) /  desviacion[iy]
        return matrixArray

    def correlation(self):
        '''
        Calcular la matriz de correlaciones
        '''
        matrix = np.dot(self.matrixArray,np.transpose(self.matrixArray))
        correlation = matrix/self.matrixArray.shape[0]
        return correlation


        

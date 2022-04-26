
import numpy as np

class myPCA:

    def __init__(self,matrixArray):
        self.matrixArray = self.convertMatrix(matrixArray)
        self.correlationMatrix = self.correlation()
        self.eigVal, self.eigVec = self.eigen()
        # Se procede a obtener la matriz C
        self.C = np.matmul(np.transpose(self.matrixArray),self.eigVec)
        self.inertia = self.getInertia()

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

    def eigen(self):
        '''
        Calcular y ordenar los eigenvector y los eigenvalues
        '''
        eigVal, eigVec = np.linalg.eigh(self.correlationMatrix)
        eigVal = np.absolute(eigVal)
        sortIndex = np.argsort(eigVal)[::-1]
        eigVal = eigVal[sortIndex]
        eigVec = np.transpose(eigVec)
        eigVec = eigVec[sortIndex]
        eigVec = np.transpose(eigVec)
        return eigVal,eigVec
    
    def getInertia(self):
        '''
        Obtener la matriz de correlación
        '''
        totalValues = self.correlationMatrix.shape[0]
        return [value/totalValues for value in self.eigVal]

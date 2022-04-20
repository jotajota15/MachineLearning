# -*- coding: utf-8 -*-

# Bibliotecas

import math
import random

import matplotlib.pyplot as plt
import numpy as np

##### Graficos simples usando listas

plt.plot([1, 2, 3, 4])
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()

# color: r,g,b,c,m,y,w,k  red,green,blue,cyan,magenta,yellow,white,black + '#00FF00' (HEX color)
# tipo de punto: . o ^ v > < s p * + x D
# tipo de linea: - -- : -.

plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'r*-.')
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'r*-.')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()

plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16],'r*-.')
plt.xlabel('Eje X')
plt.xlim([0,5])
plt.ylabel('Eje Y')
plt.ylim([0,20])
plt.show()

##### Ploteo funciones

ejex = []
x = [] # f(x) = x
x2 = [] # f(x) = x^2
x3 = [] # f(x) = x^3

for i in range(0,301):
 	n = i/100 # [0,300] de 1 en 1, [0,3] de 0.01 en 0.01
 	ejex.append(n)
 	x.append(n)
 	x2.append(n**2)
 	x3.append(n**3)
 	
plt.plot(ejex,x,'r-')
plt.plot(ejex,x2,'g-')
plt.plot(ejex,x3,'b-')
plt.show()

x = []
y = []
def seno_cardinal(x):
 	return math.sin(math.pi*x)/(math.pi*x) if x!=0 else 0

for i in np.arange(-9,9,0.01):
 	x.append( i )
 	y.append( seno_cardinal(i) )
 	
plt.plot(x,y,'b-')
plt.title("Seno cardinal")
plt.show()

x = []
y = []
for i in np.arange(0,math.pi*2,0.01):
    x.append(i)
    y.append((math.cos(1*i)+1)/2) # Intervalo de [-1,1] +1 -> [0,2] /2 -> [0,1]

plt.polar(x,y)
plt.show()

##### Histogramas

def tirar_dado(caras):
 	return int(random.random()*caras + 1)

tiradas = []

for i in range(36000):
 	dos_dados = tirar_dado(6)+tirar_dado(6)
 	tiradas.append(dos_dados)

# 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
etiquetas = [str(i) for i in range(2,13)]

plt.hist(tiradas, bins=11, label=etiquetas)
plt.xlabel("Suma de tirar dos dados de seis caras (2D6)")
plt.show()

##### Scatter plot

# Abre el archivo en modo lectura
archivo = open('iris.data','r')
# Lee las lineas no vacias, remueve los saltos de linea y separa por comas
m = [l.replace("\n","").split(",") for l in archivo.readlines() if l.replace("\n","")!=""]
# Convierte los primeros 4 valores de cada fila en flotantes, forma la matriz
m = [[float(data[0]),float(data[1]),float(data[2]),float(data[3]),data[4]] for data in m]
archivo.close()

# Extrae la columna col de la matriz
def columna(matriz,col):
 	columna = []
 	for f in range(len(matriz)):
  		columna.append(matriz[f][col])
 	return columna

# Genera un arreglo de "colores" conforme a la especie de la flor
colors = []
for f in range(len(m)):
 	if m[f][4]=="Iris-setosa":
  		colors.append('g')
 	elif m[f][4]=="Iris-versicolor":
  		colors.append('r')
 	else:
  		colors.append('b')

plt.scatter(columna(m,0),columna(m,1),c = colors)
plt.show()

plt.scatter(columna(m,0),columna(m,2),c = colors)
plt.show()

plt.scatter(columna(m,0),columna(m,3),c = colors)
plt.show()

plt.scatter(columna(m,1),columna(m,2),c = colors)
plt.show()

plt.scatter(columna(m,1),columna(m,3),c = colors)
plt.show()

plt.scatter(columna(m,2),columna(m,3),c = colors)
plt.show()

##### Usando pandas

import pandas as pd

# Creando el dataframe de una matriz en memoria
df = pd.DataFrame(m,columns=['sepal_length','sepal_width','petal_length','petal_width','flower_type'])

# O cargar el dataframe directamente del archivo
df2 = pd.read_csv('iris.data', names=['sepal_length','sepal_width','petal_length','petal_width','flower_type'])

import seaborn as sns
# Nos permite hacer el ploteo en todas las columnas
sns.pairplot(df,hue='flower_type')

### Probemos con otros datos: SAHeart.csv un compilado de datos sobre pacientes con posible enfermedad coronaria en Sudafrica

df = pd.read_csv("SAheart.csv",delimiter=";",decimal=".")
print("Tamaño de datos: ",df.shape)
print("Columnas: ",df.columns)
print("Primeros 5 elementos:\n", df.head(5))

#sbp
##    systolic blood pressure
#tobacco
##    cumulative tobacco (kg)
#ldl
##    low density lipoprotein cholesterol
#adiposity
##    a numeric vector
#famhist
##    family history of heart disease, a factor with levels Absent Present
#typea
##    type-A behavior
#obesity
##    a numeric vector
#alcohol
##    current alcohol consumption
#age
##    age at onset
#chd
##    response, coronary heart disease

print("Tipos de datos:",df.dtypes)

# Tenemos datos numericos y categoricos

numericos = df.loc[:,['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']]
categoricos = df.loc[:,['famhist','chd']]

# Podemos realizar analisis estadistico sobre los datos de tipo numerico

print("Descripcion:",numericos.describe())

# Y a los categoricos podemos analizarles su distribucion

for column in categoricos.columns:
	datos = categoricos.loc[:,column]
	categorias = datos.unique()
	mapeo = {}
	for i in range(len(categorias)):
		mapeo[categorias[i]] = i
	plt.hist(datos.map(mapeo),bins=len(categorias))
	plt.title("Distribucion de los datos de "+column)
	plt.xlabel("Tipo")
	plt.ylabel("Cantidad")
	plt.xticks(range(len(categorias)),categorias)
	plt.show()
    
# Tambien podemos graficar diferentes datos numéricos según nuestro interes
	
plt.hist(numericos['obesity'],bins=30)
plt.show()

# Sin embargo, Python no se lleva bien con los datos categóricos
## Para hacer ploteos como el scatter plot necesitamos convertilos en números
## O para calcular correlaciones

# Get dummies convierte cada categoria en un grupo de columnas propia
# Cada columna representa una categoria y presenta un 0 o un 1, dependiendo de la pertenencia
# Esto se conoce como "One-hot encoding"
df = pd.get_dummies(df,columns=['chd','famhist'])

print(df.head(5))

# Ahora que tenemos los datos, veamos como se ve un scatter plot de todos contra todos
## Nota: Usaremos el valor de 'chd' -> la presencia de enfermedad coronaria para colorear los puntos

sns.pairplot(df,hue='chd_Si')

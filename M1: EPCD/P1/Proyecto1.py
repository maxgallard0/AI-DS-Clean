# 1. Importación de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### 2. Cargar y Preparar los Datos

# Cargar los datos desde el archivo .txt
data = pd.read_csv('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/P1/Proyecto 1 Simple Linear Regression Data.txt', header=None, names=['Population', 'Profit'])

# Separar los datos en X y Y
x = data['Population'].values.reshape(-1, 1)
y = data['Profit'].values.reshape(-1, 1)

# Agregar una columna de 1s para el vector de intersección (bias)
m = len(y)
x = np.c_[np.ones(m), x]

# 3. Función para graficar los datos

# Definir la función para graficar los datos y la recta de regresión
def graficaDatos(x, y, theta=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', marker='o', label='Data')
    plt.xlabel('Población de la ciudad en 10Ks')
    plt.ylabel('Profit en $10Ks')
    
    if theta is not None:
        plt.plot(x, np.dot(np.c_[np.ones(x.shape[0]), x], theta), label='Regresión lineal', color='red')
    
    plt.legend()
    plt.grid(True)
    plt.show()

graficaDatos(x[:, 1], y)

# 4. Función para calcular el costo (Función de error)

# Definir la función de coste
def calculaCosto(x, y, theta):
    m = len(y)
    J = np.sum((np.dot(x, theta) - y) ** 2) / (2 * m)
    return J

# Calcular costo inicial
initial_cost = calculaCosto(x, y, theta=np.zeros((2, 1)))
initial_cost

# 5. Función de descenso por gradiente

# Definir la función de gradiente descendente
def gradienteDescendente(X, y, theta, alpha, iteraciones):
    m = len(y)
    J_history = []
    
    for i in range(iteraciones):
        theta = theta - (alpha / m) * np.dot(X.T, (np.dot(X, theta) - y))
        J_history.append(calculaCosto(x, y, theta))
    
    return theta, J_history

# 6. Ejecutar el modelo

# Ejecutar descenso por gradiente
alpha = 0.01
iteraciones = 1500
theta, J_history = gradienteDescendente(x, y, np.zeros((2, 1)), alpha, iteraciones)
theta.flatten()

# 7. Realizar predicciones

prediccion1 = np.dot([1, 3.5], theta)[0]
prediccion2 = np.dot([1, 7], theta)[0]
prediccion1, prediccion2

# 8. Graficar resultados

graficaDatos(x[:, 1], y, theta)
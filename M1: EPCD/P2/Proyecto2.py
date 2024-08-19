import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def graficaDatos(X, y, theta=None):
    # Separar los ejemplos admitidos y no admitidos
    admitidos = y == 1
    no_admitidos = y == 0

    # Graficar los ejemplos sin normalización
    plt.scatter(X[admitidos, 0], X[admitidos, 1], c='blue', marker='x', label='Admitidos')
    plt.scatter(X[no_admitidos, 0], X[no_admitidos, 1], c='red', marker='o', label='No Admitidos')

    # Graficar la recta de decisión si se proporciona theta
    if theta is not None:
        # Graficar en la escala original (0 a 100)
        x_values = np.array([0, 100])
        # Desnormalizar theta para graficar correctamente
        theta_0 = theta[0] - np.sum((theta[1:] * scaler.mean_) / scaler.scale_)
        theta_1 = theta[1] / scaler.scale_[0]
        theta_2 = theta[2] / scaler.scale_[1]
        y_values = -(theta_0 + theta_1 * x_values) / theta_2
        plt.plot(x_values, y_values, label='Límite de decisión')

    # Configuración de la gráfica
    plt.xlim(0, 100)  # Limitar el eje X de 0 a 100
    plt.ylim(0, 100)  # Limitar el eje Y de 0 a 100
    plt.xlabel('Examen 1')
    plt.ylabel('Examen 2')
    plt.legend()
    plt.show()

def sigmoidal(z):
    return 1 / (1 + np.exp(-z))

def funcionCosto(theta, X, y):
    m = len(y)
    h = sigmoidal(np.dot(X, theta))
    J = -(1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    grad = (1/m) * np.dot(X.T, (h - y))
    return J, grad

def aprende(theta, X, y, iteraciones, alpha):
    for i in range(iteraciones):
        J, grad = funcionCosto(theta, X, y)
        theta -= alpha * grad
    return theta

def predice(theta, X):
    probabilidad = sigmoidal(np.dot(X, theta))
    return (probabilidad >= 0.5).astype(int)

# Cargar los datos
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]  # No normalizar estos datos para graficar
y = data[:, 2]

# Mantener una copia de los datos originales para graficar
X_original = X.copy()

# Normalizar características para el entrenamiento (excluyendo la columna de 1's)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Agregar columna de 1's a X para el entrenamiento
m = X.shape[0]
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# Inicializar theta
theta = np.zeros(X.shape[1])

# Definir parámetros
iteraciones = 10000  # Aumentar el número de iteraciones
alpha = 0.001  # Ajustar la tasa de aprendizaje

# Entrenar el modelo
theta_optimo = aprende(theta, X, y, iteraciones, alpha)

# Mostrar theta óptimo
print(f'Theta óptimo: {theta_optimo}')

# Graficar datos y recta de decisión utilizando los datos originales
graficaDatos(X_original, y, theta_optimo)

# Predicción para un nuevo estudiante (usamos el modelo entrenado con datos normalizados)
nuevo_estudiante = np.array([45, 85])
nuevo_estudiante_norm = scaler.transform([nuevo_estudiante])
nuevo_estudiante_norm = np.concatenate([np.ones(1), nuevo_estudiante_norm[0]])

prediccion = predice(theta_optimo, nuevo_estudiante_norm)
print(f'Predicción para el nuevo estudiante: {"Admitido" if prediccion == 1 else "No Admitido"}')
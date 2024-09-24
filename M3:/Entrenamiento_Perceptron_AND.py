import numpy as np

# Parámetros iniciales
learning_rate = 0.1  # Tasa de aprendizaje (η)
bias = 1            # Valor del bias
weights = np.random.rand(3)  # Inicialización de los pesos [w0, w1, w2] de forma aleatoria

# Datos de entrada AND (x1, x2) con bias
X = np.array([
    [bias, 1, 1],
    [bias, 1, -1],
    [bias, -1, 1],
    [bias, -1, -1]
])

# Salidas deseadas para la función AND
y_desired = np.array([1, -1, -1, -1])

# Función de activación escalón binario
def step_function(x):
    return 1 if x >= 0 else -1

# Entrenamiento del perceptrón
epoch = 0
while True:
    epoch += 1
    print(f"\nEpoch {epoch}")
    error_total = 0  # Variable para acumular el error total en la época

    for i in range(len(X)):
        # Cálculo de la salida del perceptrón
        y = step_function(np.dot(weights, X[i]))

        # Error entre la salida deseada y la obtenida
        error = y_desired[i] - y
        error_total += abs(error)  # Acumular el error total

        # Actualización de pesos
        weights += learning_rate * error * X[i]

        # Mostrar resultados
        print(f"Entrada: {X[i][1:]}, Salida deseada: {y_desired[i]}, Salida obtenida: {y}, Pesos: {weights}, Error: {error}")

    # Si el error total es 0, significa que todos los ejemplos fueron clasificados correctamente
    if error_total == 0:
        break

print(f"\nPesos finales: {weights}")
import numpy as np

# Clase Adaline
class Adaline:
    def __init__(self, input_size, learning_rate=0.01, initial_weights=None):
        if initial_weights is None:
            self.weights = np.random.rand(input_size)  # Pesos aleatorios si no se proporcionan
        else:
            self.weights = np.array(initial_weights)  # Pesos iniciales específicos
        self.learning_rate = learning_rate
    
    def predict(self, x):
        return np.dot(x, self.weights)
    
    def train_until_convergence(self, X, y, max_iterations=1000000, tolerance=1):
        iterations = 0
        while iterations < max_iterations:
            output = self.predict(X)
            errors = y - output
            self.weights += self.learning_rate * np.dot(X.T, errors)
            cost = (errors**2).sum() / 2.0
            iterations += 1
            if cost < tolerance:
                return iterations
        return -1  # No convergió

# Función para mostrar resultados de entrenamiento
def show_training_results(logic_type, X, y):
    adaline.weights = np.array(initial_weights)  # Reiniciar los pesos
    iterations = adaline.train_until_convergence(X, y)
    if iterations == -1:
        print(f"\nError: No se pudo obtener convergencia para {logic_type}.")
    else:
        print(f"\nIteraciones necesarias para {logic_type}: {iterations}")
        print(f"Pesos finales para {logic_type}: {adaline.weights}")

# Inicialización de pesos específicos
initial_weights = [1.5, 0.5, 1.5]

# Crear el modelo Adaline con pesos iniciales específicos y alpha fijado en 0.01
adaline = Adaline(input_size=3, learning_rate=0.01, initial_weights=initial_weights)

# Entrenar y mostrar resultados para cada función lógica

# Función AND
X_and = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y_and = np.array([0, 0, 0, 1])
show_training_results("AND", X_and, y_and)

# Función OR
X_or = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y_or = np.array([0, 1, 1, 1])
show_training_results("OR", X_or, y_or)

# Función NOR
X_nor = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y_nor = np.array([1, 0, 0, 0])
show_training_results("NOR", X_nor, y_nor)

# Función para realizar predicciones según los valores ingresados por el usuario
def make_prediction(adaline, logic_type):
    print(f"\nRealizando predicción para {logic_type}:")
    x1 = int(input("Introduce el valor de x1 (0 o 1): "))
    x2 = int(input("Introduce el valor de x2 (0 o 1): "))
    x = np.array([1, x1, x2])  # Agregar el sesgo (bias)
    output = adaline.predict(x)
    result = 1 if output >= 0.5 else 0  # Adaline produce salida continua, se convierte a binaria
    print(f"La salida para {logic_type} con entradas ({x1}, {x2}) es: {result}")

# Solicitar valores al usuario y hacer predicciones
make_prediction(adaline, "OR")
make_prediction(adaline, "AND")
make_prediction(adaline, "NOR")
import numpy as np

# Función escalón binaria
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptrón
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, initial_weights=None):
        if initial_weights is None:
            self.weights = np.random.rand(input_size)  # Pesos aleatorios si no se proporcionan
        else:
            self.weights = np.array(initial_weights)  # Pesos iniciales específicos
        self.learning_rate = learning_rate
    
    def predict(self, x):
        weighted_sum = np.dot(x, self.weights)
        return step_function(weighted_sum)
    
    def train_until_convergence(self, X, y, max_iterations=1000):
        iterations = 0
        while iterations < max_iterations:
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                total_error += abs(error)
                self.weights += self.learning_rate * error * X[i]
            iterations += 1
            # Verificar si el entrenamiento debe detenerse (es decir, no hay errores)
            if total_error == 0:
                return iterations
        # Si se alcanzó el número máximo de iteraciones sin converger
        return -1

def load_data_from_file(filename):
    data = np.loadtxt(filename)
    X = np.c_[np.ones(data.shape[0]), data[:, :-1]]  # Agregar columna de 1's para x0
    y = data[:, -1]
    return X, y

# Inicialización de pesos específicos
initial_weights = [1.5, 0.5, 1.5]

# Crear el perceptrón con pesos iniciales específicos y alpha fijado en 0.1
perceptron = Perceptron(input_size=3, learning_rate=0.1, initial_weights=initial_weights)

# Función para mostrar resultados de entrenamiento
def show_training_results(logic_type, X, y):
    perceptron.weights = np.array(initial_weights)  # Reiniciar los pesos
    iterations = perceptron.train_until_convergence(X, y)
    if iterations == -1:
        print(f"\nError: No se pudo obtener convergencia para {logic_type}.")
    else:
        print(f"\nIteraciones necesarias para {logic_type}: {iterations}")
        print(f"Pesos finales para {logic_type}: {perceptron.weights}")

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
def make_prediction(perceptron, logic_type):
    print(f"\nRealizando predicción para {logic_type}:")
    x1 = int(input("Introduce el valor de x1 (0 o 1): "))
    x2 = int(input("Introduce el valor de x2 (0 o 1): "))
    x = np.array([1, x1, x2])  # Agregar el sesgo (bias)
    result = perceptron.predict(x)
    print(f"La salida para {logic_type} con entradas ({x1}, {x2}) es: {result}")

# Solicitar valores al usuario y hacer predicciones
make_prediction(perceptron, "OR")
make_prediction(perceptron, "AND")
make_prediction(perceptron, "NOR")
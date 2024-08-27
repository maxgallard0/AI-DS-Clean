import numpy as np

# Función de activación Sigmoide, que introduce no linealidad en la red
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradiente de la función Sigmoide, necesario para backpropagation
def sigmoidalGradiente(z):
    g = sigmoid(z)
    return g * (1 - g)

# Inicialización aleatoria de pesos para romper la simetría
def randInicializacionPesos(L_in, L_out):
    epsilon_init = 0.12  # Tamaño del rango para inicializar pesos, evita valores cercanos a 0
    return np.random.rand(L_out, L_in) * 2 * epsilon_init - epsilon_init

# Función para calcular el costo y los gradientes
def calcularCostoGradientes(W1, b1, W2, b2, X, y, reg_param):
    m = X.shape[0]  # Número de ejemplos de entrenamiento
    
    # Convertir las etiquetas en una matriz de one-hot encoding para cálculo del costo
    y_matrix = np.eye(W2.shape[0])[y.flatten() - 1] 
    
    # Forward propagation
    Z1 = np.dot(X, W1.T) + b1.T
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2.T) + b2.T
    A2 = sigmoid(Z2)
    
    # Cálculo del costo con regularización
    costo = (-1/m) * np.sum(y_matrix * np.log(A2) + (1 - y_matrix) * np.log(1 - A2))
    costo += (reg_param/(2*m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    # Backpropagation
    dZ2 = A2 - y_matrix
    dW2 = (1/m) * np.dot(dZ2.T, A1) + (reg_param/m) * W2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2) * sigmoidalGradiente(Z1)
    dW1 = (1/m) * np.dot(dZ1.T, X) + (reg_param/m) * W1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    return costo, dW1, db1, dW2, db2

# Función para verificar el gradiente
def verificarGradiente(W1, b1, W2, b2, X, y, reg_param, epsilon=1e-4):
    # Obtener gradientes analíticos
    _, dW1, db1, dW2, db2 = calcularCostoGradientes(W1, b1, W2, b2, X, y, reg_param)
    
    # Inicializar gradientes numéricos
    dW1_num = np.zeros(W1.shape)
    dW2_num = np.zeros(W2.shape)
    db1_num = np.zeros(b1.shape)
    db2_num = np.zeros(b2.shape)
    
    # Verificación del gradiente para W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_pos = np.copy(W1)
            W1_neg = np.copy(W1)
            W1_pos[i, j] += epsilon
            W1_neg[i, j] -= epsilon
            
            costo_pos, _, _, _, _ = calcularCostoGradientes(W1_pos, b1, W2, b2, X, y, reg_param)
            costo_neg, _, _, _, _ = calcularCostoGradientes(W1_neg, b1, W2, b2, X, y, reg_param)
            
            dW1_num[i, j] = (costo_pos - costo_neg) / (2 * epsilon)
    
    # Verificación del gradiente para W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_pos = np.copy(W2)
            W2_neg = np.copy(W2)
            W2_pos[i, j] += epsilon
            W2_neg[i, j] -= epsilon
            
            costo_pos, _, _, _, _ = calcularCostoGradientes(W1, b1, W2_pos, b2, X, y, reg_param)
            costo_neg, _, _, _, _ = calcularCostoGradientes(W1, b1, W2_neg, b2, X, y, reg_param)
            
            dW2_num[i, j] = (costo_pos - costo_neg) / (2 * epsilon)
    
    # Verificación del gradiente para b1
    for i in range(b1.shape[0]):
        b1_pos = np.copy(b1)
        b1_neg = np.copy(b1)
        b1_pos[i, 0] += epsilon
        b1_neg[i, 0] -= epsilon
        
        costo_pos, _, _, _, _ = calcularCostoGradientes(W1, b1_pos, W2, b2, X, y, reg_param)
        costo_neg, _, _, _, _ = calcularCostoGradientes(W1, b1_neg, W2, b2, X, y, reg_param)
        
        db1_num[i, 0] = (costo_pos - costo_neg) / (2 * epsilon)
    
    # Verificación del gradiente para b2
    for i in range(b2.shape[0]):
        b2_pos = np.copy(b2)
        b2_neg = np.copy(b2)
        b2_pos[i, 0] += epsilon
        b2_neg[i, 0] -= epsilon
        
        costo_pos, _, _, _, _ = calcularCostoGradientes(W1, b1, W2, b2_pos, X, y, reg_param)
        costo_neg, _, _, _, _ = calcularCostoGradientes(W1, b1, W2, b2_neg, X, y, reg_param)
        
        db2_num[i, 0] = (costo_pos - costo_neg) / (2 * epsilon)
    
    # Comparación de gradientes
    diff_W1 = np.linalg.norm(dW1 - dW1_num) / np.linalg.norm(dW1 + dW1_num)
    diff_W2 = np.linalg.norm(dW2 - dW2_num) / np.linalg.norm(dW2 + dW2_num)
    diff_b1 = np.linalg.norm(db1 - db1_num) / np.linalg.norm(db1 + db1_num)
    diff_b2 = np.linalg.norm(db2 - db2_num) / np.linalg.norm(db2 + db2_num)
    
    print(f"Diferencia relativa para W1: {diff_W1:.8f}")
    print(f"Diferencia relativa para W2: {diff_W2:.8f}")
    print(f"Diferencia relativa para b1: {diff_b1:.8f}")
    print(f"Diferencia relativa para b2: {diff_b2:.8f}")
    
    return diff_W1, diff_W2, diff_b1, diff_b2

# Función principal para entrenar la red neuronal
def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y, alpha, num_iter, reg_param, verificar_gradiente=False):
    # Imprimir los hiperparámetros utilizados
    print(f"Entrenando la Red Neuronal con los siguientes hiperparámetros:")
    print(f"Tasa de Aprendizaje (alpha): {alpha}")
    print(f"Número de Iteraciones: {num_iter}")
    print(f"Parámetro de Regularización (lambda): {reg_param}")
    print(f"Dimensión de Entrada: {input_layer_size}")
    print(f"Dimensión de la Capa Oculta: {hidden_layer_size}")
    print(f"Número de Clases: {num_labels}\n")

    # Inicialización de los pesos y biases
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    b1 = np.zeros((hidden_layer_size, 1))
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)
    b2 = np.zeros((num_labels, 1))
    
    # Verificación del gradiente (opcional)
    if verificar_gradiente:
        print("Verificando gradiente...")
        verificarGradiente(W1, b1, W2, b2, X, y, reg_param)
    
    m = X.shape[0]  # Número de ejemplos de entrenamiento
    
    # Convertir las etiquetas en una matriz de one-hot encoding para cálculo del costo
    y_matrix = np.eye(num_labels)[y.flatten() - 1] 
    
    for i in range(num_iter):
        # Forward propagation: calcular activaciones
        Z1 = np.dot(X, W1.T) + b1.T  # Entrada de la capa oculta
        A1 = sigmoid(Z1)  # Salida de la capa oculta (aplicación de la función sigmoide)
        Z2 = np.dot(A1, W2.T) + b2.T  # Entrada de la capa de salida
        A2 = sigmoid(Z2)  # Salida de la capa de salida, probabilidad para cada clase
        
        # Cálculo del costo con regularización
        costo = (-1/m) * np.sum(y_matrix * np.log(A2) + (1 - y_matrix) * np.log(1 - A2))
        costo += (reg_param/(2*m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        
        # Backpropagation: calcular gradientes
        dZ2 = A2 - y_matrix  # Error en la salida
        dW2 = (1/m) * np.dot(dZ2.T, A1) + (reg_param/m) * W2  # Gradiente de W2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # Gradiente de b2
        dZ1 = np.dot(dZ2, W2) * sigmoidalGradiente(Z1)  # Error en la capa oculta
        dW1 = (1/m) * np.dot(dZ1.T, X) + (reg_param/m) * W1  # Gradiente de W1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # Gradiente de b1
        
        # Actualización de pesos utilizando gradientes
        W1 -= alpha * dW1
        b1 -= alpha * db1.T
        W2 -= alpha * dW2
        b2 -= alpha * db2.T
        
        # Imprimir costo en cada múltiplo de 100 y en la última iteración
        if i % 100 == 0 or i == num_iter - 1:
            print(f"Iteración {i}: Costo = {costo}")
    
    return W1, b1, W2, b2

# Función para predecir con la red neuronal entrenada con capa oculta
def prediceRNYaEntrenada(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1.T) + b1.T  # Forward propagation para la capa oculta
    A1 = sigmoid(Z1)  # Aplicar la función de activación
    Z2 = np.dot(A1, W2.T) + b2.T  # Forward propagation para la capa de salida
    A2 = sigmoid(Z2)  # Obtener las probabilidades de cada clase
    return np.argmax(A2, axis=1) + 1  # Seleccionar la clase con mayor probabilidad

# Cargar los datos reales desde el archivo 'dígitos.txt' en la ruta especificada
def cargar_datos():
    data = np.loadtxt('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M1: EPCD/P3/digitos.txt')  # Cargar los datos desde el archivo
    X = data[:, :-1]  # Todas las columnas excepto la última (características)
    y = data[:, -1].astype(int)   # La última columna (etiquetas), convertida a enteros

    # Ajuste de etiquetas: Convertir '0' a '10' para compatibilidad con one-hot encoding
    y = np.where(y == 0, 10, y)

    print("Rango de etiquetas en el conjunto de datos:", np.min(y), "a", np.max(y))
    print("Distribución de las etiquetas:", dict(zip(*np.unique(y, return_counts=True))))
    
    # Agregar columna de 1's para el bias
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    return X, y

# Cargar datos y entrenar la red neuronal
X, y = cargar_datos()

# Definir las dimensiones de la red neuronal
input_layer_size = 400  # Número de unidades de la capa de entrada (400 píxeles)
hidden_layer_size = 25   # Número de unidades en la capa oculta
num_labels = 10          # Número de etiquetas (dígitos del 1 al 10)

# Entrenar la red neuronal con capa oculta y verificar el gradiente
W1, b1, W2, b2 = entrenaRN(input_layer_size + 1, hidden_layer_size, num_labels, X, y, alpha = 0.1, num_iter = 15000, reg_param = 0.1, verificar_gradiente=True)

# Predecir usando la red entrenada con capa oculta
y_pred = prediceRNYaEntrenada(X, W1, b1, W2, b2)

# Calcular la precisión
precision = np.mean(y_pred == y) * 100
print(f"Precisión del modelo con datos reales y capa oculta: {precision:.2f}%")
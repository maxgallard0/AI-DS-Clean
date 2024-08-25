import pandas as pd
import math
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import plotly.express as px

# Activar el comportamiento futuro para evitar downcasting silencioso
pd.set_option('future.no_silent_downcasting', True)

# Specify the file paths
cancer_test_file = '/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M2: DSACD/ACT 2/cancerTest.txt'
cancer_training_file = '/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M2: DSACD/ACT 2/cancerTraining.txt'

# Define los nombres de las columnas basados en las etiquetas proporcionadas
column_names = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Read the files into pandas dataframes without headers and with no column used as an index
cancer_test = pd.read_csv(cancer_test_file, header=None, names=column_names)
cancer_training = pd.read_csv(cancer_training_file, header=None, names=column_names)

# Reemplaza las etiquetas de clasificación en la columna 'Class'
cancer_test['Class'] = cancer_test['Class'].replace({'benign': 0.0, 'malignant': 1.0}).infer_objects()
cancer_training['Class'] = cancer_training['Class'].replace({'benign': 0.0, 'malignant': 1.0}).infer_objects()

# Implementación de funciones
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def relu(z):
    return max(0, z)

def tanh(z):
    return (2 / (1 + math.exp(-2 * z))) - 1

def gradient(sampleList, weights, activation='sigmoid'):
    sumElements = 0.0
    for x,y in zip(sampleList, weights):
        sumElements += x * y
    
    if activation == 'relu':
        return relu(sumElements)
    elif activation == 'tanh':
        return tanh(sumElements)
    else:  # sigmoid por defecto
        return sigmoid(sumElements)

def stochasticGradientAscent(trainingLists, traningLabels, featureNumber, iterations=150, activation='sigmoid'):
    sampleNumber = len(trainingLists)
    weights = [1.0] * featureNumber

    for x in range(iterations):
        sampleIndex = list(range(sampleNumber))
        for y in range(sampleNumber):
            alpha = 4/(1.0 + x + y) + 0.01
            randIndex = int(random.uniform(0, len(sampleIndex)))
            sampleGradient = gradient(trainingLists[randIndex], weights, activation)
            error = traningLabels[randIndex] - sampleGradient

            for index in range(featureNumber):
                weights[index] += alpha * error * trainingLists[randIndex][index]
            
            del(sampleIndex[randIndex])
    
    return weights

def classifyList(testList, weights, activation='sigmoid'):
    sumElements = 0
    for x, y in zip(testList, weights):
        sumElements += (x * y)
    
    probability = sigmoid(sumElements) if activation == 'sigmoid' else relu(sumElements) if activation == 'relu' else tanh(sumElements)
    
    return 1.0 if probability > 0.5 else 0.0

# Preparación de datos
X_train = cancer_training.iloc[:, :-1].values.tolist()
y_train = cancer_training['Class'].values.tolist()
X_test = cancer_test.iloc[:, :-1].values.tolist()
y_test = cancer_test['Class'].values.tolist()

# Entrenamiento del modelo manual con diferentes funciones de activación
activations = ['sigmoid', 'relu', 'tanh']
results = {}

for activation in activations:
    print(f"\nEntrenando con activación: {activation}")
    optimalWeights = stochasticGradientAscent(X_train, y_train, len(X_train[0]), activation=activation)

    predictions = [classifyList(x, optimalWeights, activation=activation) for x in X_test]

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    results[activation] = {
        'accuracy': accuracy,
        'recall': recall,
        'confusion_matrix': cm
    }

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")

    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_title(f'Matriz de confusión - {activation}')
    plt.show()

# Implementación opcional con scikit-learn
print("\nEntrenamiento con scikit-learn LogisticRegression")

X_train_array = cancer_training.iloc[:, :-1].values
y_train_array = cancer_training['Class'].values
X_test_array = cancer_test.iloc[:, :-1].values
y_test_array = cancer_test['Class'].values

# Definir el modelo de regresión logística
model = LogisticRegression()

# Definir los parámetros para grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularización
    'max_iter': [100, 150, 200],  # Número de iteraciones
}

# Configurar GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train_array, y_train_array)

# Evaluar en los datos de prueba
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_array)

accuracy = accuracy_score(y_test_array, y_pred)
recall = recall_score(y_test_array, y_pred)
cm = confusion_matrix(y_test_array, y_pred)

print(f"Mejor modelo: {best_model}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.ax_.set_title('Matriz de confusión - scikit-learn')
plt.show()

# Aplicar la regresión logística sobre los datos de prueba usando el enfoque de scikit-learn
print("\nAplicar la regresión logística sobre muestras de prueba usando scikit-learn")

# Crear el clasificador de regresión logística
logistic = LogisticRegression()

# Ajustar el modelo a los datos de entrenamiento
logistic.fit(X_train_array, y_train_array)

# Hacer predicciones sobre los datos de prueba
predictions = logistic.predict(X_test_array)

# Imprimir cada predicción con el valor real
print("Predicciones vs. Etiquetas Reales:")
for i in range(len(predictions)):
    print("Predicted: " + str(predictions[i]) + " Real Value: " + str(y_test_array[i]))

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test_array, predictions)
print("Model accuracy:", accuracy)

# Mostrar la matriz de confusión usando Plotly Express
print("\nVisualizando la matriz de confusión usando Plotly Express")

# Generar la matriz de confusión
cm = confusion_matrix(y_test_array, predictions)

# Visualizar la matriz de confusión usando Plotly Express
fig = px.imshow(cm,
    labels=dict(x="Etiquetas precedidas", y="Etiquetas verdaderas", color="Conteo"),
    x=["Clase 0", "Clase 1"],  # Reemplaza con los nombres de clases si es necesario
    y=["Clase 0", "Clase 1"],
    title="Matriz de confusión",
    color_continuous_scale="Blues")
fig.update_xaxes(side="bottom")
fig.show()

# Calcular precisión y recall para '1.0' como clase positiva
precision1 = precision_score(y_test_array, predictions, pos_label=1.0)
recall1 = recall_score(y_test_array, predictions, pos_label=1.0)

# Calcular precisión y recall para '0.0' como clase positiva
precision0 = precision_score(y_test_array, predictions, pos_label=0.0)
recall0 = recall_score(y_test_array, predictions, pos_label=0.0)

# Imprimir los resultados
print("When '1.0' is positive class:")
print("Precision:", precision1)
print("Recall:", recall1)
print("When '0.0' is positive class:")
print("Precision:", precision0)
print("Recall:", recall0)

# Crear un DataFrame para las métricas de precisión y recall
metrics_df = pd.DataFrame({
    "Class": ["0.0", "0.0", "1.0", "1.0"],
    "Metric": ["Precision", "Recall", "Precision", "Recall"],
    "Value": [precision0, recall0, precision1, recall1]
})

# Crear el gráfico de barras agrupadas
fig = px.bar(metrics_df,
             x='Class',
             y='Value',
             color='Metric',
             barmode='group',
             title="Precision and Recall Comparison for Classes 0 and 1",
             labels={'Value': 'Score', 'Class': 'Class'},
             text_auto=True)

# Mostrar el gráfico
fig.show()

# Propuesta de visualizaciones adicionales con Plotly Express
# Puedes agregar aquí cuatro visualizaciones adicionales que consideres relevantes.
# Ejemplos de visualizaciones adicionales pueden incluir:

# 1. Visualización de la distribución de las características entre las clases
fig = px.histogram(cancer_training, x="Uniformity of Cell Size", color="Class", barmode="overlay", title="Distribución de Uniformity of Cell Size por Clase")
fig.show()

# 2. Box plot de características seleccionadas por clase
fig = px.box(cancer_training, x="Class", y="Clump Thickness", color="Class", title="Distribución de Clump Thickness por Clase")
fig.show()

# 3. Scatter plot de dos características principales
fig = px.scatter(cancer_training, x="Uniformity of Cell Shape", y="Single Epithelial Cell Size", color="Class", title="Scatter Plot de Características por Clase")
fig.show()

# 4. Correlación de características usando un heatmap
corr = cancer_training.corr()
fig = px.imshow(corr, text_auto=True, title="Mapa de Calor de Correlaciones entre Características")
fig.show()

# Descripción de hallazgos y conclusión
# En esta sección, puedes proporcionar un resumen de los hallazgos clave basados en las visualizaciones anteriores
# y hacer una conclusión sobre el proceso de experimentación.
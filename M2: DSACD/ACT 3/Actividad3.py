import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold

# Cargar los datos de entrenamiento
training_texts = []
training_labels = []

with codecs.open('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M2: DSACD/ACT 3/training.txt', 'r', 'UTF-8') as file:
    for line in file:
        elements = line.split('@@@')
        training_texts.append(elements[0])
        training_labels.append(elements[1].strip())

# Cargar los datos de prueba
test_texts = []
test_labels = []

with codecs.open('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M2: DSACD/ACT 3/test.txt', 'r', 'UTF-8') as file:
    for line in file:
        elements = line.split('@@@')
        test_texts.append(elements[0])
        test_labels.append(elements[1].strip())

# Crear un vector de 'Bag of Words'
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_texts)
y_train = training_labels

X_test = vectorizer.transform(test_texts)
y_test = test_labels

# Definir las características a utilizar y K-Fold Cross Validation
feature_sizes = [20, 40, 60, 80, 100, 120]
k_values = [3, 4, 5, 6]

results = []

for features in feature_sizes:
    X_train_selected = X_train[:, :features]
    X_test_selected = X_test[:, :features]
    
    for k in k_values:
        # Configurar el modelo Naive Bayes
        model = MultinomialNB()
        
        # Validación cruzada
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='accuracy')
        
        # Entrenar y evaluar en el set de prueba
        model.fit(X_train_selected, y_train)
        accuracy = model.score(X_test_selected, y_test)
        precision = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='precision_macro').mean()
        recall = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='recall_macro').mean()
        f1 = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='f1_macro').mean()
        
        # Guardar resultados
        results.append({
            'features': features,
            'k': k,
            'cross_val_accuracy': np.mean(scores),
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

# Convertir resultados en un DataFrame para análisis
results_df = pd.DataFrame(results)

# 1. Accuracy vs Number of Features for Different K (Gráfico de Línea)
plt.figure(figsize=(10, 6))
for k in k_values:
    subset = results_df[results_df['k'] == k]
    plt.plot(subset['features'], subset['cross_val_accuracy'], marker='o', label=f'K={k} Cross-Val')
plt.title('Accuracy vs Number of Features for Different K')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 2. Precision vs Number of Features (Gráfico de Barras)
plt.figure(figsize=(10, 6))
subset = results_df.groupby('features')['precision'].mean()
subset.plot(kind='bar')
plt.title('Precision vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Precision')
plt.show()

# 3. Recall vs Number of Features for Each K (Gráfico de Línea con Múltiples Líneas)
plt.figure(figsize=(10, 6))
for k in k_values:
    subset = results_df[results_df['k'] == k]
    plt.plot(subset['features'], subset['recall'], marker='o', label=f'K={k} Cross-Val')
plt.title('Recall vs Number of Features for Different K')
plt.xlabel('Number of Features')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()

# 4. F1 Score Distribution for Each K (Gráfico de Caja y Bigotes)
plt.figure(figsize=(10, 6))
results_df.boxplot(column='f1', by='k')
plt.title('F1 Score Distribution for Each K')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.suptitle('')  # Remove the automatic suptitle
plt.show()

# 5. Heatmap of Accuracy for Features and K (Mapa de Calor)
pivot_table = results_df.pivot('features', 'k', 'cross_val_accuracy')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('Heatmap of Accuracy for Features and K')
plt.xlabel('K Value')
plt.ylabel('Number of Features')
plt.show()

# 6. Model Performance Comparison Across Metrics (Gráfico de Barras Apiladas)
plt.figure(figsize=(10, 6))
subset = results_df[results_df['features'] == 60]
bar_width = 0.25
index = np.arange(len(k_values))

plt.bar(index, subset['cross_val_accuracy'], bar_width, label='Accuracy')
plt.bar(index + bar_width, subset['precision'], bar_width, label='Precision')
plt.bar(index + 2 * bar_width, subset['recall'], bar_width, label='Recall')
plt.bar(index + 3 * bar_width, subset['f1'], bar_width, label='F1 Score')

plt.xlabel('K Value')
plt.ylabel('Score')
plt.title('Model Performance Comparison Across Metrics for 60 Features')
plt.xticks(index + bar_width, k_values)
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import os  # Para obtener la ruta del escritorio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.optimizers.legacy import Adam # type: ignore
import seaborn as sns
import joblib

# Definir la ruta para guardar las imágenes en el escritorio
desktop_path = os.path.expanduser('~/Desktop')

# Cargar los datos
df = pd.read_csv('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/Proyecto Final/Catan_Stats_S.csv')

# Preprocesamiento de datos
X = df[['settlement1_dice1', 'settlement1_dice2', 'settlement1_dice3',
        'settlement2_dice1', 'settlement2_dice2', 'settlement2_dice3',
        'num_lumber', 'num_wheat', 'num_clay', 'num_sheep', 'num_ore',
        'num_3G', 'num_2(X)', 'num_D']]
y = df['winner']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear las clases en los datos de entrenamiento
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Estandarizar los datos
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Ajuste de las clases ponderadas (class_weight) para manejar el desequilibrio
class_weight = {0: 1.0, 1: 4}

# Construcción del modelo de red neuronal con 4 capas ocultas y regularización L2
model = models.Sequential()

# Capa de entrada
model.add(layers.Dropout(0.1, input_shape=(X_train_resampled.shape[1],)))

# Primera capa oculta con regularización L2 y Dropout
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))

# Segunda capa oculta con regularización L2 y Dropout
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))

# Tercera capa oculta con regularización L2 y Dropout
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))

# Capa de salida (clasificación binaria)
model.add(layers.Dense(1, activation='sigmoid'))

# Compilación del modelo con el optimizador legacy Adam
adam_optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping para detener el entrenamiento cuando no haya mejoras
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Entrenamiento del modelo con class_weight
history = model.fit(X_train_resampled, y_train_resampled, epochs=5000, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping], class_weight=class_weight)

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Predicciones
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")  # Convertir probabilidades a etiquetas binarias

# Calcular el F1 Score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

# Métricas de rendimiento
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calcular el AUC-ROC
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Graficar la curva ROC y guardar la imagen
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Red Neuronal con 4 Capas Ocultas, SMOTE y Class Weight')
plt.legend(loc="lower right")
plt.savefig(f'{desktop_path}/roc_curve.png')  # Guardar en el escritorio
plt.show()

# Graficar la pérdida (loss) de entrenamiento y validación y guardar la imagen
plt.figure(figsize=(10, 5))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Exactitud (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.savefig(f'{desktop_path}/training_validation_loss_accuracy.png')  # Guardar en el escritorio
plt.show()

# Imprimir AUC
print(f'ROC AUC: {roc_auc:.2f}')

### Graficar la Matriz de Confusión ###
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Ganará', 'Ganará'], yticklabels=['No Ganará', 'Ganará'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Matriz de Confusión')
plt.savefig(f'{desktop_path}/confusion_matrix.png')  # Guardar en el escritorio
plt.show()

# Guardar el scaler después de ajustarlo
scaler_filename = '/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/Proyecto Final/scaler.pkl'
joblib.dump(scaler, scaler_filename)

# Guardar el modelo como un archivo .h5
model.save('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/Proyecto Final/Catan_Model.h5')

### Guardar el reporte de clasificación como imagen ###

# Generar el reporte de clasificación
classification_report_str = classification_report(y_test, y_pred, target_names=['No Ganará', 'Ganará'])

# Crear una figura para el reporte de clasificación
plt.figure(figsize=(8, 6))
plt.text(0.01, 1.25, str('Reporte de Clasificación'), {'fontsize': 12}, fontproperties='monospace')  # Título
plt.text(0.01, 0.05, str(classification_report_str), {'fontsize': 10}, fontproperties='monospace')  # Reporte en formato de texto
plt.axis('off')  # Ocultar los ejes
plt.savefig(f'{desktop_path}/classification_report.png')  # Guardar en el escritorio
plt.show()
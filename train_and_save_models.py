import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.patches as mpatches

# ------------------------------------------
timings = {}
start_time = time.time()

print("📂 Cargando base de datos...")
data = pd.read_csv('diabetes_database.csv')

# Altura en cm
if data['Altura'].max() < 10:
    data['Altura'] = data['Altura'] * 100

# Limpia Sexo
data['Sexo'] = data['Sexo'].astype(str).str.strip().str.capitalize()
data['Sexo'] = data['Sexo'].map({'Mujer': 0, 'Hombre': 1})

# Limpia Antecedentes
if data['Antecedentes'].dtype == 'object':
    data['Antecedentes'] = data['Antecedentes'].map({'Yes': 1, 'No': 0})

print("📊 Variables:", data.columns)

# Correlación
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('correlation_matrix_v2.png')
plt.show()

X = data.drop('Salida', axis=1).values
y = data['Salida'].values

timings['Carga y limpieza'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balancear con SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

timings['Preprocesamiento'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

# SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
svm_model = grid.best_estimator_
print(f"✅ Mejor SVM: {grid.best_params_}")

svm_train_decision = svm_model.decision_function(X_train_scaled)
svm_test_decision = svm_model.decision_function(X_test_scaled)

timings['Entrenamiento SVM'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_train_proba = rf.predict_proba(X_train_scaled)[:, 1]
rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]

timings['Entrenamiento RF'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

# NN con SVM + RF
X_train_combined = np.column_stack((X_train_scaled, svm_train_decision, rf_train_proba))
X_test_combined = np.column_stack((X_test_scaled, svm_test_decision, rf_test_proba))

model = Sequential([
    Dense(32, input_dim=X_train_combined.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

history = model.fit(
    X_train_combined, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

timings['Entrenamiento NN'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

y_pred_combined = (model.predict(X_test_combined) > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred_combined)
auc = roc_auc_score(y_test, y_pred_combined)
f1 = f1_score(y_test, y_pred_combined)

print(f"\n✅ Precisión: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred_combined)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Modelo Combinado')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('confusion_matrix_v2.png')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Pérdida')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves_v2.png')
plt.show()

timings['Evaluación'] = time.time() - start_time

# ------------------------------------------
start_time = time.time()

# Análisis conjunto: SVM + RF + NN
decision_scores = svm_model.decision_function(X_test_scaled)
proba_scores_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
proba_scores_rf = rf.predict_proba(X_test_scaled)[:, 1]
nn_outputs = model.predict(X_test_combined).flatten()

df_results = pd.DataFrame({
    'Decision_Score_SVM': decision_scores,
    'Probability_SVM': proba_scores_svm,
    'Probability_RF': proba_scores_rf,
    'NN_Output': nn_outputs,
    'True_Label': y_test
})

legend_labels = {0: 'Non Diabetic', 1: 'Diabetic'}
handles = [mpatches.Patch(color=plt.cm.coolwarm(i/1), label=label) for i, label in enumerate(legend_labels.values())]

# 1) Decision Score SVM vs Probability SVM
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_results['Decision_Score_SVM'],
    df_results['Probability_SVM'],
    c=df_results['True_Label'],
    cmap='coolwarm',
    alpha=0.8,
    edgecolors='black',
    s=60
)
plt.title('Decision Score vs Probability SVM')
plt.xlabel('Decision Score SVM')
plt.ylabel('Probability SVM')
plt.legend(handles=handles, title="Classes")
plt.colorbar(scatter, label='True Class')
plt.grid(True, linestyle='--')
plt.savefig('decision_vs_probability_svm.png')
plt.show()

# 2) Probability SVM vs Probability RF
plt.figure(figsize=(10, 6))
scatter2 = plt.scatter(
    df_results['Probability_SVM'],
    df_results['Probability_RF'],
    c=df_results['True_Label'],
    cmap='coolwarm',
    alpha=0.8,
    edgecolors='black',
    s=60
)
plt.title('Probability SVM vs Probability RF')
plt.xlabel('Probability SVM')
plt.ylabel('Probability RF')
plt.legend(handles=handles, title="Classes")
plt.colorbar(scatter2, label='True Class')
plt.grid(True, linestyle='--')
plt.savefig('probability_svm_vs_rf.png')
plt.show()


plt.figure(figsize=(10, 6))
scatter3 = plt.scatter(
    df_results['Probability_RF'],
    df_results['NN_Output'],
    c=df_results['True_Label'],
    cmap='coolwarm',
    alpha=0.8,
    edgecolors='black',
    s=60
)
plt.title('Probability RF vs NN Output')
plt.xlabel('Probability RF')
plt.ylabel('NN Output')
plt.legend(handles=handles, title="Classes")
plt.colorbar(scatter3, label='True Class')
plt.grid(True, linestyle='--')
plt.savefig('probability_rf_vs_nnoutput.png')
plt.show()

# 4) 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter4 = ax.scatter(
    df_results['Probability_SVM'],
    df_results['Probability_RF'],
    df_results['NN_Output'],
    c=df_results['True_Label'],
    cmap='coolwarm',
    alpha=0.8,
    edgecolors='black',
    s=60
)
ax.set_xlabel('Probability SVM')
ax.set_ylabel('Probability RF')
ax.set_zlabel('NN Output')
ax.set_title('3D: SVM + RF + NN')
ax.view_init(elev=25, azim=45)
plt.colorbar(scatter4, shrink=0.6)
plt.savefig('3d_svm_rf_nn.png')
plt.show()

acc = accuracy_score(y_test, y_pred_combined)
auc = roc_auc_score(y_test, y_pred_combined)
f1 = f1_score(y_test, y_pred_combined)

print(f"\n✅ Precisión: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

timings['Análisis conjunto'] = time.time() - start_time
metrics = {
    'Precisión': acc,
    'AUC': auc,
    'F1 Score': f1
}

plt.figure(figsize=(8,5))
bars = plt.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylim(0, 1.05)

# Mostrar valor encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', fontsize=12)

plt.title('Métricas de desempeño del modelo combinado', fontsize=14)
plt.ylabel('Valor')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('metricas_desempeno.png')
plt.show()



# ------------------------------------------
# Guardar modelos
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(svm_model, 'models/svm_model.joblib')
joblib.dump(rf, 'models/rf_model.joblib')
model.save('models/nn_model.h5')
print("\n✅ Modelos guardados en carpeta 'models/'")

print("\n⏱️ Tiempos de ejecución:")
for k, v in timings.items():
    print(f"{k}: {v:.2f} s") 
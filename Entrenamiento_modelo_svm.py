import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Archivos a guardar
MODEL_FILE = os.path.join(BASE_DIR, "modelo_svm.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_FILE = os.path.join(BASE_DIR, "label_encoders.pkl")
COLUMNS_FILE = os.path.join(BASE_DIR, "columns.pkl")

# ===============================
# 1️⃣ Eliminar PKL viejos si existen
# ===============================
for f in [MODEL_FILE, SCALER_FILE, ENCODERS_FILE, COLUMNS_FILE]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Eliminado archivo antiguo: {f}")

# ===============================
# 2️⃣ Cargar y preprocesar datos
# ===============================
df = pd.read_csv(os.path.join(BASE_DIR, "bank-full.csv"), sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Columnas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# LabelEncoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, ENCODERS_FILE)
print("✅ LabelEncoders guardados")

# Separar X e y
X = df.drop('y', axis=1)
y = df['y']

# Escalar columnas numéricas
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
joblib.dump(scaler, SCALER_FILE)
print("✅ Scaler guardado")

# Guardar columnas finales (orden correcto)
columns_finales = X.columns.tolist()
joblib.dump(columns_finales, COLUMNS_FILE)
print("✅ Columnas finales guardadas")

# ===============================
# 3️⃣ Separar train/test para evaluación
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4️⃣ Entrenar modelo SVM
# ===============================
model = SVC(probability=True, kernel='rbf', random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_FILE)
print("✅ Modelo SVM entrenado y guardado")

# ===============================
# 5️⃣ Evaluar modelo
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n📊 Métricas del modelo:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

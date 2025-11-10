import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from tqdm import tqdm  # <-- barra de progreso

# ====== Cargar CSV completo ======
print("Cargando CSV...")
df = pd.read_csv("bank_sample.csv", sep=';')
print(f"✅ CSV cargado ({len(df)} filas)")

# ====== Crear subset balanceado de 50k datos ======
print("Creando subset balanceado de 50k filas...")
n_total = 50000
pos_df = df[df['y']=='yes']
neg_df = df[df['y']=='no']

ratio_pos = len(pos_df) / len(df)
n_pos = int(n_total * ratio_pos)
n_neg = n_total - n_pos

pos_sample = pos_df.sample(n=n_pos, random_state=42)
neg_sample = neg_df.sample(n=n_neg, random_state=42)

df_sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
print(f"✅ Subset creado ({len(df_sample)} filas)")

# ====== Separar features y target ======
X = df_sample.drop('y', axis=1)
y = df_sample['y']

cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
num_cols = ['age','balance','day','duration','campaign','pdays','previous']

# ====== LabelEncoder para variables categóricas ======
print("Codificando variables categóricas...")
label_encoders = {}
for i, col in enumerate(tqdm(cat_cols, desc="Categorical columns")):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Target encoder
y_le = LabelEncoder()
y = y_le.fit_transform(y)  # 0/1
print("✅ Variables categóricas codificadas")

# ====== Escalado ======
print("Escalando variables numéricas...")
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print("✅ Escalado completado")

# ====== Split train/test ======
print("Separando datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Datos listos (Train: {len(X_train)}, Test: {len(X_test)})")

# ====== Entrenamiento SVM ======
print("Entrenando modelo SVM (kernel linear, balanceado)...")
svm_model = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)

# Mostrar barra de progreso “simulada” porque SVC.fit no da updates por defecto
for i in tqdm(range(1), desc="SVM Training"):
    svm_model.fit(X_train, y_train)

print("✅ Entrenamiento completado")

# ====== Evaluación rápida ======
y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)[:,1]

print("==== Métricas del modelo ====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ====== Guardar modelos para dashboard ======
print("Guardando modelos...")
joblib.dump(svm_model, "modelo_svm.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(y_le, "target_encoder.pkl")
print("✅ Modelos entrenados y guardados correctamente")

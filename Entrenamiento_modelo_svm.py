import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# 1️⃣ Eliminar modelos antiguos
# ===============================
archivos = ["modelo_rf.pkl", "scaler.pkl", "label_encoders.pkl", "columnas_esperadas.pkl"]
for f in archivos:
    ruta = os.path.join(BASE_DIR, f)
    if os.path.exists(ruta):
        os.remove(ruta)
        print(f"🗑️ Eliminado: {f}")

# ===============================
# 2️⃣ Cargar datos
# ===============================
df = pd.read_csv(os.path.join(BASE_DIR, "bank-full.csv"), sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# ===============================
# 3️⃣ Separar columnas categóricas y numéricas
# ===============================
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = ['age','balance','day','duration','campaign','pdays','previous']

# ===============================
# 4️⃣ Codificar categorías
# ===============================
encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

# ===============================
# 5️⃣ Escalar columnas numéricas
# ===============================
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# ===============================
# 6️⃣ Separar X e y
# ===============================
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Guardar columnas esperadas
columnas_esperadas = X.columns.tolist()
joblib.dump(columnas_esperadas, os.path.join(BASE_DIR, "columnas_esperadas.pkl"))

# ===============================
# 7️⃣ Entrenar RandomForest
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 8️⃣ Evaluar
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy RandomForest: {acc:.4f}")

# ===============================
# 9️⃣ Guardar artefactos
# ===============================
joblib.dump(model, os.path.join(BASE_DIR, "modelo_rf.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(BASE_DIR, "label_encoders.pkl"))

print("✅ Entrenamiento completado. Artefactos guardados: modelo_rf.pkl, scaler.pkl, label_encoders.pkl, columnas_esperadas.pkl")

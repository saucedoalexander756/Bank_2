# verificar_pkls.py
import os
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(".")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_svm.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
COLUMNAS_PATH = os.path.join(BASE_DIR, "columnas_esperadas.pkl")
CSV_PATH = os.path.join(BASE_DIR, "bank-full.csv")

print("=== Verificando archivos .pkl ===\n")

# 1️⃣ Modelo
if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)
    print(f"[OK] modelo_svm.pkl -> {type(modelo)}")
else:
    print("[FALTA] modelo_svm.pkl")

# 2️⃣ Scaler
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] scaler.pkl -> {type(scaler)}")
else:
    print("[FALTA] scaler.pkl")

# 3️⃣ Label Encoders
if os.path.exists(ENCODERS_PATH):
    encoders = joblib.load(ENCODERS_PATH)
    print(f"[OK] label_encoders.pkl -> {type(encoders)}")
else:
    print("[FALTA] label_encoders.pkl")

# 4️⃣ Columnas esperadas
if os.path.exists(COLUMNAS_PATH):
    columnas_esperadas = joblib.load(COLUMNAS_PATH)
    print(f"[OK] columnas_esperadas.pkl -> {len(columnas_esperadas)} columnas")
else:
    print("[FALTA] columnas_esperadas.pkl")

# 5️⃣ Comprobar columnas actuales del CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, sep=';')
    df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=False)
    print(f"\n[INFO] Columnas actuales del CSV (one-hot): {len(df_encoded.columns)} columnas")
    
    if os.path.exists(COLUMNAS_PATH):
        # Comparar columnas
        csv_cols = set(df_encoded.columns)
        expected_cols = set(columnas_esperadas)
        faltan = expected_cols - csv_cols
        extras = csv_cols - expected_cols

        print(f"[INFO] Columnas faltantes para el modelo: {len(faltan)} -> {faltan}" if faltan else "[OK] No faltan columnas")
        print(f"[INFO] Columnas extra no esperadas: {len(extras)} -> {extras}" if extras else "[OK] No hay columnas extra")
else:
    print("[FALTA] bank-full.csv")

print("\n✅ Verificación completa")

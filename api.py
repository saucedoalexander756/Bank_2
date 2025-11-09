from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay,
    ConfusionMatrixDisplay, PrecisionRecallDisplay
)
# NUEVA IMPLEMENTACIÓN: Librería para trabajar con SQLite
import sqlite3
# NUEVA IMPLEMENTACIÓN: Librería para manejar el tiempo
from datetime import datetime

# Inicializar la aplicación FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# Variables Globales y Rutas
# =======================================================
accuracy = None
precision = None
recall = None
f1 = None
roc_auc = None
cm = None
initialization_error = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_svm.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl") #nuevo

# NUEVA IMPLEMENTACIÓN: Ruta de la base de datos SQLite
DB_PATH = os.path.join(BASE_DIR, "svm_metrics.db")

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFUSION_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
ROC_PATH = os.path.join(PLOTS_DIR, "roc_curve.png")
PR_PATH = os.path.join(PLOTS_DIR, "precision_recall_curve.png")

# =======================================================
# NUEVA IMPLEMENTACIÓN: Funciones de Base de Datos
# =======================================================

def initialize_db():
    """Crea la tabla de historial de métricas si no existe."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metric_history (
            timestamp TEXT PRIMARY KEY,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL
        )
    """)
    conn.commit()
    conn.close()

def record_metrics(acc, prec, rec, f1s, roc):
    """Guarda las métricas actuales en la tabla de historial."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO metric_history (timestamp, accuracy, precision, recall, f1_score, roc_auc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, acc, prec, rec, f1s, roc))
        conn.commit()
        print(f"Métricas registradas en la DB: {timestamp}")
    except sqlite3.IntegrityError:
        print("Advertencia: Intento de insertar registro duplicado (misma marca de tiempo).")
    finally:
        conn.close()

# Inicializar la base de datos al inicio
initialize_db()

# =======================================================
# Lógica de Inicialización: Cargando modelos, datos y precalculando métricas
# =======================================================
try:
    # ... (Lógica de carga de modelos y preprocesamiento existente) ...
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modelo o Scaler no encontrados.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_raw = pd.read_csv(os.path.join(BASE_DIR, "bank-full.csv"), sep=';')
    
    df_raw['y'] = df_raw['y'].map({'yes': 1, 'no': 0})
    
    categorical_cols_to_encode = df_raw.select_dtypes(include=['object']).columns.tolist()
    df_processed = pd.get_dummies(df_raw, columns=categorical_cols_to_encode, drop_first=True, dtype=int)
    
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']

    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    X_scaled_df = X
    
    y_pred = model.predict(X_scaled_df)
    y_prob = model.predict_proba(X_scaled_df)[:, 1]

    # Almacenar métricas globales
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    # NUEVA IMPLEMENTACIÓN: Registrar las métricas en la base de datos
    record_metrics(accuracy, precision, recall, f1, roc_auc)

    # ========================
    # Generar gráficas (Lógica existente)
    # ========================
    plt.style.use('dark_background')
    
    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(ax=ax)
    plt.title("Matriz de Confusión - SVM")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Curva ROC
    fig, ax = plt.subplots(figsize=(8,6))
    RocCurveDisplay.from_estimator(model, X_scaled_df, y, ax=ax)
    plt.title("Curva ROC/AUC - SVM")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Curva Precisión-Recall
    fig, ax = plt.subplots(figsize=(8,6))
    PrecisionRecallDisplay.from_estimator(model, X_scaled_df, y, ax=ax)
    plt.title("Curva Precisión vs Recall - SVM")
    plt.tight_layout()
    plt.savefig(PR_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Inicialización de modelos, métricas y registro histórico completado con éxito.")

except Exception as e:
    initialization_error = str(e)
    print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN DE ML: {initialization_error}")


# ========================
# Endpoints API (Añadiendo /history)
# ========================
@app.get("/")
def root():
    if initialization_error:
        return {"message": "API de Clasificación SVM iniciada, pero las métricas no están disponibles debido a un error.", "error": initialization_error}
    return {"message": "API de Clasificación SVM con métricas y gráficas"}

@app.get("/metrics")
def get_metrics():
    if initialization_error:
        print("💥 initialization_error:", initialization_error)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error al inicializar las métricas. Verifique los archivos o versiones de Scikit-learn.",
                "details": initialization_error

            }
        )
    
    tn, fp, fn, tp = cm.ravel()
    
    return JSONResponse({
        "Modelo": "Support Vector Machine (SVM)",
        "Accuracy": float(round(accuracy, 4)),
        "Precision": float(round(precision, 4)),
        "Recall": float(round(recall, 4)), 
        "F1_Score": float(round(f1, 4)),
        "ROC_AUC": float(round(roc_auc, 4)),
        "Confusion_Matrix": np.array(cm).tolist(),
        "tn": int(tn), 
        "fp": int(fp), 
        "fn": int(fn), 
        "tp": int(tp)  
    })

# NUEVO ENDPOINT: Para obtener el historial de métricas
@app.get("/history")
def get_history():
    """Devuelve todos los registros históricos de métricas de la base de datos."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Usar row_factory para obtener resultados como diccionarios/filas
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM metric_history ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()

        # Convertir las filas (objetos Row) en una lista de diccionarios
        history_data = [dict(row) for row in rows]

        if not history_data:
            return JSONResponse({"message": "No hay registros históricos disponibles."}, status_code=200)

        return JSONResponse(history_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Error al recuperar el historial de la base de datos.", "details": str(e)}
        )

# El resto de los endpoints de plots (sin cambios)
@app.get("/plot/confusion")
def get_confusion_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error": "Gráfica no generada debido al error de inicialización."})
    return FileResponse(CONFUSION_PATH)

@app.get("/plot/roc")
def get_roc_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error": "Gráfica no generada debido al error de inicialización."})
    return FileResponse(ROC_PATH)

@app.get("/plot/precision_recall")
def get_precision_recall_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error": "Gráfica no generada debido al error de inicialización."})
    return FileResponse(PR_PATH)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
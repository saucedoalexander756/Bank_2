from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay,
    ConfusionMatrixDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from datetime import datetime

# ===============================
# Inicializar FastAPI
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===============================
# Variables globales
# ===============================
accuracy = precision = recall = f1 = roc_auc = None
cm = None
initialization_error = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_svm.pkl")       # RandomForest
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columnas_esperadas.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

DB_PATH = os.path.join(BASE_DIR, "rf_metrics.db")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFUSION_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
ROC_PATH = os.path.join(PLOTS_DIR, "roc_curve.png")
PR_PATH = os.path.join(PLOTS_DIR, "precision_recall_curve.png")

# ===============================
# Base de datos
# ===============================
def initialize_db():
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO metric_history (timestamp, accuracy, precision, recall, f1_score, roc_auc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, acc, prec, rec, f1s, roc))
        conn.commit()
    finally:
        conn.close()

initialize_db()

# ===============================
# Inicialización modelo + datos
# ===============================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columnas_esperadas = joblib.load(COLUMNS_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    df_raw = pd.read_csv(os.path.join(BASE_DIR, "bank-full.csv"), sep=';')
    df_raw['y'] = df_raw['y'].map({'yes':1,'no':0})

    # Aplicar LabelEncoder
    for col, le in encoders.items():
        if col in df_raw.columns:
            valores_validos = set(le.classes_)
            df_raw[col] = df_raw[col].astype(str).apply(lambda x: x if x in valores_validos else list(valores_validos)[0])
            df_raw[col] = le.transform(df_raw[col].astype(str))

    # Preparar X
    X = df_raw.drop('y', axis=1)
    for col in columnas_esperadas:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_esperadas]

    numeric_cols = ['age','balance','day','duration','campaign','pdays','previous']
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    y = df_raw['y']
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:,1]
    except AttributeError:
        probs = model.predict(X)
        y_prob = MinMaxScaler().fit_transform(probs.reshape(-1,1)).flatten()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    record_metrics(accuracy, precision, recall, f1, roc_auc)

    # Gráficas
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(ax=ax)
    plt.title("Matriz de Confusión - RF")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
    plt.title("Curva ROC/AUC - RF")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    PrecisionRecallDisplay.from_predictions(y, y_prob, ax=ax)
    plt.title("Curva Precisión vs Recall - RF")
    plt.tight_layout()
    plt.savefig(PR_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

except Exception as e:
    initialization_error = str(e)
    print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN: {initialization_error}")

# ===============================
# Endpoints FastAPI
# ===============================
@app.get("/")
def root():
    if initialization_error:
        return {"message":"API iniciada con errores", "error": initialization_error}
    return {"message":"API de Clasificación RF con métricas y gráficas"}

@app.get("/metrics")
def get_metrics():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error":initialization_error})
    tn, fp, fn, tp = cm.ravel()
    return JSONResponse({
        "Modelo":"RandomForest",
        "Accuracy": round(accuracy,4),
        "Precision": round(precision,4),
        "Recall": round(recall,4),
        "F1_Score": round(f1,4),
        "ROC_AUC": round(roc_auc,4),
        "Confusion_Matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    })

@app.get("/history")
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metric_history ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()
        return JSONResponse([dict(row) for row in rows])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e)})

@app.get("/plot/confusion")
def get_confusion_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error":initialization_error})
    return FileResponse(CONFUSION_PATH)

@app.get("/plot/roc")
def get_roc_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error":initialization_error})
    return FileResponse(ROC_PATH)

@app.get("/plot/precision_recall")
def get_precision_recall_plot():
    if initialization_error:
        return JSONResponse(status_code=500, content={"error":initialization_error})
    return FileResponse(PR_PATH)

# ===============================
# Ejecutar API
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


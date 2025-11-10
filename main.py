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
MODEL_PATH = os.path.join(BASE_DIR, "modelo_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columnas_esperadas.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "bank-full.csv")

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

# ===============================
# Inicialización modelo + datos (MODIFICADO)
# ===============================
def initialize_app():
    global accuracy, precision, recall, f1, roc_auc, cm, initialization_error
    
    try:
        # Verificar que los archivos existan
        required_files = {
            "modelo": MODEL_PATH,
            "scaler": SCALER_PATH, 
            "columnas": COLUMNS_PATH,
            "encoders": ENCODERS_PATH,
            "datos": DATA_PATH
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name} ({path})")
        
        if missing_files:
            raise FileNotFoundError(f"Archivos faltantes: {', '.join(missing_files)}")

        # Si todos los archivos existen, cargarlos
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        columnas_esperadas = joblib.load(COLUMNS_PATH)
        encoders = joblib.load(ENCODERS_PATH)

        df_raw = pd.read_csv(DATA_PATH, sep=';')
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

        # Inicializar BD después de verificar archivos
        initialize_db()
        record_metrics(accuracy, precision, recall, f1, roc_auc)

        # Gráficas
        plt.style.use('default')  # Cambiar a tema por defecto

        fig, ax = plt.subplots(figsize=(8,6))
        ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(ax=ax)
        plt.title("Matriz de Confusión - RF")
        plt.tight_layout()
        plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
        plt.title("Curva ROC/AUC - RF")
        plt.tight_layout()
        plt.savefig(ROC_PATH, dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        PrecisionRecallDisplay.from_predictions(y, y_prob, ax=ax)
        plt.title("Curva Precisión vs Recall - RF")
        plt.tight_layout()
        plt.savefig(PR_PATH, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("✅ Aplicación inicializada correctamente")
        
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ ERROR durante la inicialización: {initialization_error}")

# Inicializar la aplicación al importar
initialize_app()

# ===============================
# Endpoints FastAPI
# ===============================
@app.get("/")
def root():
    if initialization_error:
        return {
            "message": "API iniciada con errores", 
            "error": initialization_error,
            "status": "error"
        }
    return {
        "message": "API de Clasificación RF con métricas y gráficas",
        "status": "running"
    }

@app.get("/metrics")
def get_metrics():
    if initialization_error:
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error}
        )
    
    if cm is not None:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return JSONResponse({
        "Modelo": "RandomForest",
        "Accuracy": round(accuracy, 4) if accuracy else 0,
        "Precision": round(precision, 4) if precision else 0,
        "Recall": round(recall, 4) if recall else 0,
        "F1_Score": round(f1, 4) if f1 else 0,
        "ROC_AUC": round(roc_auc, 4) if roc_auc else 0,
        "Confusion_Matrix": cm.tolist() if cm is not None else [],
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
        
        if not rows:
            return JSONResponse({"message": "No hay registros históricos disponibles."})
            
        return JSONResponse([dict(row) for row in rows])
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/plot/confusion")
def get_confusion_plot():
    if initialization_error or not os.path.exists(CONFUSION_PATH):
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error or "Gráfica no disponible"}
        )
    return FileResponse(CONFUSION_PATH)

@app.get("/plot/roc")
def get_roc_plot():
    if initialization_error or not os.path.exists(ROC_PATH):
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error or "Gráfica no disponible"}
        )
    return FileResponse(ROC_PATH)

@app.get("/plot/precision_recall")
def get_precision_recall_plot():
    if initialization_error or not os.path.exists(PR_PATH):
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error or "Gráfica no disponible"}
        )
    return FileResponse(PR_PATH)

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not initialization_error else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "initialization_error": initialization_error
    }


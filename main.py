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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
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
using_demo_model = False  # Nueva variable para trackear modo demo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_rf.pkl")       # RandomForest
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
# Función para crear modelo demo
# ===============================
def create_demo_model():
    """Crear modelo y datos de demostración cuando los archivos reales fallan"""
    print("🔄 Creando modelo de demostración...")
    
    # Generar datos sintéticos
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_redundant=2, 
        n_informative=8,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Entrenar modelo de demo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Hacer predicciones
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    
    print("✅ Modelo de demostración creado exitosamente")
    return model, X, y, y_pred, y_prob, accuracy, precision, recall, f1, roc_auc, cm

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
            roc_auc REAL,
            model_type TEXT
        )
    """)
    conn.commit()
    conn.close()

def record_metrics(acc, prec, rec, f1s, roc, model_type="real"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO metric_history (timestamp, accuracy, precision, recall, f1_score, roc_auc, model_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, acc, prec, rec, f1s, roc, model_type))
        conn.commit()
    finally:
        conn.close()

initialize_db()

# ===============================
# Inicialización modelo + datos
# ===============================
try:
    # Intentar cargar archivos reales
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

    record_metrics(accuracy, precision, recall, f1, roc_auc, "real")
    using_demo_model = False

except Exception as e:
    print(f"❌ Error cargando modelo real: {e}")
    print("🔄 Usando modelo de demostración...")
    
    # Crear modelo demo
    model, X, y, y_pred, y_prob, accuracy, precision, recall, f1, roc_auc, cm = create_demo_model()
    record_metrics(accuracy, precision, recall, f1, roc_auc, "demo")
    using_demo_model = True
    initialization_error = f"Modelo real no disponible. Usando demo. Error: {str(e)}"

# Generar gráficas (funciona para ambos modos)
try:
    plt.style.use('default')  # Cambiar a tema por defecto para mejor compatibilidad

    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(ax=ax)
    plt.title(f"Matriz de Confusión - RF ({'DEMO' if using_demo_model else 'REAL'})")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
    plt.title(f"Curva ROC/AUC - RF ({'DEMO' if using_demo_model else 'REAL'})")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    PrecisionRecallDisplay.from_predictions(y, y_prob, ax=ax)
    plt.title(f"Curva Precisión vs Recall - RF ({'DEMO' if using_demo_model else 'REAL'})")
    plt.tight_layout()
    plt.savefig(PR_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("✅ Gráficas generadas exitosamente")

except Exception as e:
    print(f"❌ Error generando gráficas: {e}")
    if initialization_error:
        initialization_error += f" | Error gráficas: {str(e)}"
    else:
        initialization_error = f"Error generando gráficas: {str(e)}"

# ===============================
# Endpoints FastAPI
# ===============================
@app.get("/")
def root():
    if initialization_error:
        return {
            "message": "API iniciada con errores", 
            "error": initialization_error,
            "model_type": "demo" if using_demo_model else "real",
            "status": "warning" if using_demo_model else "error"
        }
    return {
        "message": "API de Clasificación RF con métricas y gráficas",
        "model_type": "demo" if using_demo_model else "real",
        "status": "running"
    }

@app.get("/metrics")
def get_metrics():
    if initialization_error and not using_demo_model:
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error}
        )
    
    tn, fp, fn, tp = cm.ravel() if cm is not None else (0, 0, 0, 0)
    
    return JSONResponse({
        "Modelo": "RandomForest",
        "Model_Type": "demo" if using_demo_model else "real",
        "Accuracy": round(accuracy, 4) if accuracy else 0,
        "Precision": round(precision, 4) if precision else 0,
        "Recall": round(recall, 4) if recall else 0,
        "F1_Score": round(f1, 4) if f1 else 0,
        "ROC_AUC": round(roc_auc, 4) if roc_auc else 0,
        "Confusion_Matrix": cm.tolist() if cm is not None else [],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "note": "Usando modelo de demostración" if using_demo_model else "Modelo real cargado"
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
        
        history_data = [dict(row) for row in rows]
        return JSONResponse({
            "data": history_data,
            "current_model_type": "demo" if using_demo_model else "real"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/plot/confusion")
def get_confusion_plot():
    if initialization_error and not using_demo_model:
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error}
        )
    
    if not os.path.exists(CONFUSION_PATH):
        return JSONResponse(
            status_code=404, 
            content={"error": "Gráfica no disponible"}
        )
    
    return FileResponse(CONFUSION_PATH)

@app.get("/plot/roc")
def get_roc_plot():
    if initialization_error and not using_demo_model:
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error}
        )
    
    if not os.path.exists(ROC_PATH):
        return JSONResponse(
            status_code=404, 
            content={"error": "Gráfica no disponible"}
        )
    
    return FileResponse(ROC_PATH)

@app.get("/plot/precision_recall")
def get_precision_recall_plot():
    if initialization_error and not using_demo_model:
        return JSONResponse(
            status_code=500, 
            content={"error": initialization_error}
        )
    
    if not os.path.exists(PR_PATH):
        return JSONResponse(
            status_code=404, 
            content={"error": "Gráfica no disponible"}
        )
    
    return FileResponse(PR_PATH)

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not initialization_error or using_demo_model else "unhealthy",
        "model_type": "demo" if using_demo_model else "real",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat(),
        "initialization_error": initialization_error
    }

@app.get("/debug")
def debug_info():
    """Endpoint para información de debug"""
    return {
        "files_exist": {
            "modelo_rf.pkl": os.path.exists(MODEL_PATH),
            "scaler.pkl": os.path.exists(SCALER_PATH),
            "columnas_esperadas.pkl": os.path.exists(COLUMNS_PATH),
            "label_encoders.pkl": os.path.exists(ENCODERS_PATH),
            "bank-full.csv": os.path.exists(DATA_PATH)
        },
        "using_demo_model": using_demo_model,
        "initialization_error": initialization_error,
        "plots_exist": {
            "confusion_matrix": os.path.exists(CONFUSION_PATH),
            "roc_curve": os.path.exists(ROC_PATH),
            "precision_recall": os.path.exists(PR_PATH)
        }
    }

# ===============================
# Ejecutar API
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


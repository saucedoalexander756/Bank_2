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
import traceback

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
model_loaded = False

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
    try:
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
        print("✅ Base de datos inicializada")
    except Exception as e:
        print(f"⚠️  Error inicializando BD: {e}")

def record_metrics(acc, prec, rec, f1s, roc):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO metric_history (timestamp, accuracy, precision, recall, f1_score, roc_auc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, acc, prec, rec, f1s, roc))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️  Error guardando métricas: {e}")

# ===============================
# Crear datos de demostración
# ===============================
def create_demo_data():
    """Crear datos de demostración si no existen los archivos reales"""
    print("🔄 Creando datos de demostración...")
    
    # Generar datos sintéticos
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_redundant=2, 
        n_informative=8,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Crear DataFrame simulado
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Entrenar modelo de demostración
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Calcular métricas
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    return model, X, y, y_pred, y_prob, df

# ===============================
# Inicialización modelo + datos
# ===============================
def initialize_app():
    global accuracy, precision, recall, f1, roc_auc, cm, initialization_error, model_loaded
    
    try:
        # Verificar archivos
        files_exist = all([
            os.path.exists(MODEL_PATH),
            os.path.exists(SCALER_PATH), 
            os.path.exists(COLUMNS_PATH),
            os.path.exists(ENCODERS_PATH),
            os.path.exists(DATA_PATH)
        ])
        
        if files_exist:
            print("✅ Cargando modelo real desde archivos...")
            # CARGAR MODELO REAL
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
                    df_raw[col] = df_raw[col].astype(str).apply(
                        lambda x: x if x in valores_validos else list(valores_validos)[0]
                    )
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
                
            model_loaded = True
            print("✅ Modelo real cargado exitosamente")
            
        else:
            print("⚠️  Usando modelo de demostración...")
            # USAR MODELO DE DEMOSTRACIÓN
            model, X, y, y_pred, y_prob, _ = create_demo_data()
            model_loaded = False

        # Calcular métricas (funciona para ambos casos)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)

        # Inicializar BD y guardar métricas
        initialize_db()
        record_metrics(accuracy, precision, recall, f1, roc_auc)

        # Generar gráficas
        plt.style.use('default')

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
        print(traceback.format_exc())

# Inicializar la aplicación
initialize_app()

# ===============================
# Endpoints FastAPI
# ===============================
@app.get("/")
def root():
    status_info = {
        "message": "API de Clasificación RF con métricas y gráficas",
        "status": "running",
        "model_loaded": model_loaded,
        "mode": "REAL" if model_loaded else "DEMO"
    }
    
    if initialization_error:
        status_info.update({
            "status": "error",
            "error": initialization_error
        })
    
    return status_info

@app.get("/metrics")
def get_metrics():
    if initialization_error:
        return JSONResponse(
            status_code=500, 
            content={
                "error": initialization_error,
                "mode": "DEMO" if not model_loaded else "REAL"
            }
        )
    
    if cm is not None:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return JSONResponse({
        "Modelo": "RandomForest",
        "Mode": "REAL" if model_loaded else "DEMO",
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
            return JSONResponse({
                "message": "No hay registros históricos disponibles.",
                "mode": "DEMO" if not model_loaded else "REAL"
            })
            
        history_data = [dict(row) for row in rows]
        return JSONResponse({
            "data": history_data,
            "mode": "DEMO" if not model_loaded else "REAL"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "mode": "DEMO" if not model_loaded else "REAL"
            }
        )

@app.get("/plot/confusion")
def get_confusion_plot():
    if initialization_error or not os.path.exists(CONFUSION_PATH):
        return JSONResponse(
            status_code=500, 
            content={
                "error": initialization_error or "Gráfica no disponible",
                "mode": "DEMO" if not model_loaded else "REAL"
            }
        )
    return FileResponse(CONFUSION_PATH)

@app.get("/plot/roc")
def get_roc_plot():
    if initialization_error or not os.path.exists(ROC_PATH):
        return JSONResponse(
            status_code=500, 
            content={
                "error": initialization_error or "Gráfica no disponible",
                "mode": "DEMO" if not model_loaded else "REAL"
            }
        )
    return FileResponse(ROC_PATH)

@app.get("/plot/precision_recall")
def get_precision_recall_plot():
    if initialization_error or not os.path.exists(PR_PATH):
        return JSONResponse(
            status_code=500, 
            content={
                "error": initialization_error or "Gráfica no disponible",
                "mode": "DEMO" if not model_loaded else "REAL"
            }
        )
    return FileResponse(PR_PATH)

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not initialization_error else "unhealthy",
        "mode": "DEMO" if not model_loaded else "REAL",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "initialization_error": initialization_error
    }

@app.get("/debug/files")
def debug_files():
    """Endpoint para debug - ver qué archivos existen"""
    files = {
        "modelo_rf.pkl": os.path.exists(MODEL_PATH),
        "scaler.pkl": os.path.exists(SCALER_PATH),
        "columnas_esperadas.pkl": os.path.exists(COLUMNS_PATH),
        "label_encoders.pkl": os.path.exists(ENCODERS_PATH),
        "bank-full.csv": os.path.exists(DATA_PATH),
        "working_directory": BASE_DIR
    }
    return files


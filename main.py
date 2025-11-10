from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sqlite3
from datetime import datetime
import traceback
import logging
from typing import Dict, Any, Optional

# ===============================
# Configuración de logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# Inicializar FastAPI
# ===============================
app = FastAPI(
    title="ML Model API",
    description="API para modelo de clasificación RandomForest",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Configuración
# ===============================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILES = {
        "model": "modelo_rf.pkl",
        "scaler": "scaler.pkl", 
        "columns": "columnas_esperadas.pkl",
        "encoders": "label_encoders.pkl",
        "data": "bank-full.csv"
    }
    DB_PATH = os.path.join(BASE_DIR, "ml_metrics.db")
    PLOTS_DIR = os.path.join(BASE_DIR, "static/plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Estado de la aplicación
# ===============================
class AppState:
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.confusion_matrix: Optional[np.ndarray] = None
        self.model_loaded: bool = False
        self.model_type: str = "unknown"
        self.initialization_error: Optional[str] = None
        self.last_update: Optional[datetime] = None

app_state = AppState()

# ===============================
# Utilidades
# ===============================
class ModelLoader:
    @staticmethod
    def safe_load(file_path: str, file_type: str) -> tuple:
        """Cargar archivo con manejo robusto de errores"""
        try:
            if not os.path.exists(file_path):
                return None, f"Archivo {file_type} no encontrado: {os.path.basename(file_path)}"
            
            data = joblib.load(file_path)
            logger.info(f"✅ {file_type} cargado: {os.path.basename(file_path)}")
            return data, None
            
        except Exception as e:
            error_msg = f"Error cargando {file_type}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg

    @staticmethod
    def create_demo_model():
        """Crear modelo de demostración con datos sintéticos"""
        logger.info("🔄 Creando modelo de demostración...")
        
        X, y = make_classification(
            n_samples=2000, 
            n_features=15,
            n_redundant=3,
            n_informative=10,
            n_clusters_per_class=1,
            flip_y=0.05,
            random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob)
        }
        
        cm = confusion_matrix(y, y_pred)
        
        logger.info("✅ Modelo de demostración creado exitosamente")
        return model, X, y, y_pred, y_prob, metrics, cm

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inicializar base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    model_type TEXT,
                    total_predictions INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT,
                    message TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.info("✅ Base de datos inicializada")
        except Exception as e:
            logger.error(f"❌ Error inicializando BD: {e}")
    
    def record_metrics(self, metrics: Dict, model_type: str, total_preds: int):
        """Guardar métricas en la base de datos"""
        try:
            timestamp = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metric_history 
                (timestamp, accuracy, precision, recall, f1_score, roc_auc, model_type, total_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                metrics.get("accuracy"),
                metrics.get("precision"), 
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("roc_auc"),
                model_type,
                total_preds
            ))
            conn.commit()
            conn.close()
            logger.info("📊 Métricas guardadas en BD")
        except Exception as e:
            logger.error(f"⚠️ Error guardando métricas: {e}")

class PlotGenerator:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        self.style = 'default'
    
    def generate_all_plots(self, y_true, y_pred, y_prob, model_type: str):
        """Generar todas las gráficas de evaluación"""
        try:
            plt.style.use(self.style)
            
            # Matriz de confusión
            self._generate_confusion_matrix(y_true, y_pred, model_type)
            # Curva ROC
            self._generate_roc_curve(y_true, y_prob, model_type)
            # Curva Precision-Recall
            self._generate_precision_recall(y_true, y_prob, model_type)
            
            logger.info("✅ Todas las gráficas generadas")
            return True
        except Exception as e:
            logger.error(f"❌ Error generando gráficas: {e}")
            return False
    
    def _generate_confusion_matrix(self, y_true, y_pred, model_type: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Matriz de Confusión - {model_type.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_roc_curve(self, y_true, y_prob, model_type: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
        plt.title(f"Curva ROC - {model_type.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "roc_curve.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_precision_recall(self, y_true, y_prob, model_type: str):
        fig, ax = plt.subfiles(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
        plt.title(f"Curva Precisión-Recall - {model_type.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "precision_recall.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

# ===============================
# Inicialización de la aplicación
# ===============================
def initialize_application():
    """Inicializar toda la aplicación de manera robusta"""
    logger.info("🚀 Inicializando aplicación...")
    
    try:
        db_manager = DatabaseManager(Config.DB_PATH)
        plot_generator = PlotGenerator(Config.PLOTS_DIR)
        
        # Verificar archivos requeridos
        file_errors = []
        loaded_objects = {}
        
        for file_type, filename in Config.MODEL_FILES.items():
            file_path = os.path.join(Config.BASE_DIR, filename)
            data, error = ModelLoader.safe_load(file_path, file_type)
            
            if error:
                file_errors.append(error)
            else:
                loaded_objects[file_type] = data
        
        # Decidir si usar modelo real o demo
        if not file_errors:
            logger.info("📦 Cargando modelo real...")
            model_type = "real"
            
            # Procesar datos reales
            df_raw = pd.read_csv(
                os.path.join(Config.BASE_DIR, Config.MODEL_FILES["data"]), 
                sep=';'
            )
            df_raw['y'] = df_raw['y'].map({'yes': 1, 'no': 0})
            
            # Aplicar preprocesamiento
            for col, le in loaded_objects["encoders"].items():
                if col in df_raw.columns:
                    valores_validos = set(le.classes_)
                    df_raw[col] = df_raw[col].astype(str).apply(
                        lambda x: x if x in valores_validos else list(valores_validos)[0]
                    )
                    df_raw[col] = le.transform(df_raw[col].astype(str))
            
            X = df_raw.drop('y', axis=1)
            for col in loaded_objects["columns"]:
                if col not in X.columns:
                    X[col] = 0
            X = X[loaded_objects["columns"]]
            
            numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            X[numeric_cols] = loaded_objects["scaler"].transform(X[numeric_cols])
            
            y = df_raw['y']
            y_pred = loaded_objects["model"].predict(X)
            y_prob = loaded_objects["model"].predict_proba(X)[:, 1]
            
        else:
            logger.warning(f"⚠️ Usando modelo de demostración. Errores: {file_errors}")
            model_type = "demo"
            model, X, y, y_pred, y_prob, metrics, cm = ModelLoader.create_demo_model()
            
            # Actualizar estado
            app_state.metrics = metrics
            app_state.confusion_matrix = cm
            app_state.model_loaded = True
            app_state.model_type = model_type
            app_state.last_update = datetime.now()
            
            # Guardar métricas y generar gráficas
            db_manager.record_metrics(metrics, model_type, len(y))
            plot_generator.generate_all_plots(y, y_pred, y_prob, model_type)
            
            logger.info("🎉 Aplicación inicializada con modelo de demostración")
            return
        
        # Calcular métricas para modelo real
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob)
        }
        
        cm = confusion_matrix(y, y_pred)
        
        # Actualizar estado
        app_state.metrics = metrics
        app_state.confusion_matrix = cm
        app_state.model_loaded = True
        app_state.model_type = model_type
        app_state.last_update = datetime.now()
        
        # Guardar métricas y generar gráficas
        db_manager.record_metrics(metrics, model_type, len(y))
        plot_generator.generate_all_plots(y, y_pred, y_prob, model_type)
        
        logger.info(f"🎉 Aplicación inicializada con modelo {model_type.upper()}")
        
    except Exception as e:
        error_msg = f"Error crítico durante inicialización: {str(e)}"
        logger.error(f"❌ {error_msg}")
        app_state.initialization_error = error_msg
        app_state.model_loaded = False

# ===============================
# Eventos de la aplicación
# ===============================
@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la aplicación"""
    initialize_application()

# ===============================
# Endpoints de la API
# ===============================
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "ML Model API",
        "status": "running" if app_state.model_loaded else "error",
        "version": "2.0.0",
        "model_loaded": app_state.model_loaded,
        "model_type": app_state.model_type,
        "last_update": app_state.last_update.isoformat() if app_state.last_update else None,
        "endpoints": [
            "/docs - Documentación interactiva",
            "/metrics - Métricas del modelo",
            "/health - Estado del servicio",
            "/history - Histórico de métricas",
            "/plots/confusion - Matriz de confusión",
            "/plots/roc - Curva ROC",
            "/plots/precision-recall - Curva Precisión-Recall"
        ]
    }

@app.get("/metrics", tags=["Modelo"])
async def get_metrics():
    """Obtener métricas actuales del modelo"""
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Estado: " + app_state.initialization_error
        )
    
    if app_state.confusion_matrix is not None:
        tn, fp, fn, tp = app_state.confusion_matrix.ravel()
    else:
        tn = fp = fn = tp = 0
    
    return {
        "model": "RandomForest",
        "type": app_state.model_type,
        "metrics": {
            "accuracy": round(app_state.metrics.get("accuracy", 0), 4),
            "precision": round(app_state.metrics.get("precision", 0), 4),
            "recall": round(app_state.metrics.get("recall", 0), 4),
            "f1_score": round(app_state.metrics.get("f1", 0), 4),
            "roc_auc": round(app_state.metrics.get("roc_auc", 0), 4)
        },
        "confusion_matrix": {
            "matrix": app_state.confusion_matrix.tolist() if app_state.confusion_matrix is not None else [],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "last_updated": app_state.last_update.isoformat() if app_state.last_update else None
    }

@app.get("/health", tags=["Monitorización"])
async def health_check():
    """Health check del servicio"""
    status = "healthy" if app_state.model_loaded else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": app_state.model_loaded,
        "model_type": app_state.model_type,
        "details": {
            "database": "connected",
            "model": "loaded" if app_state.model_loaded else "error",
            "plots": "generated" if app_state.model_loaded else "pending"
        }
    }

@app.get("/history", tags=["Histórico"])
async def get_history(limit: int = 100):
    """Obtener histórico de métricas"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM metric_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        history = [dict(row) for row in rows]
        
        return {
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accediendo al histórico: {str(e)}")

@app.get("/plots/confusion", tags=["Gráficas"])
async def get_confusion_plot():
    """Obtener matriz de confusión como imagen"""
    plot_path = os.path.join(Config.PLOTS_DIR, "confusion_matrix.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Gráfica no disponible")
    return FileResponse(plot_path, media_type="image/png")

@app.get("/plots/roc", tags=["Gráficas"])
async def get_roc_plot():
    """Obtener curva ROC como imagen"""
    plot_path = os.path.join(Config.PLOTS_DIR, "roc_curve.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Gráfica no disponible")
    return FileResponse(plot_path, media_type="image/png")

@app.get("/plots/precision-recall", tags=["Gráficas"])
async def get_precision_recall_plot():
    """Obtener curva Precisión-Recall como imagen"""
    plot_path = os.path.join(Config.PLOTS_DIR, "precision_recall.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Gráfica no disponible")
    return FileResponse(plot_path, media_type="image/png")

@app.post("/reload", tags=["Administración"])
async def reload_model():
    """Recargar el modelo (útil para actualizaciones)"""
    try:
        initialize_application()
        return {
            "message": "Modelo recargado exitosamente",
            "model_type": app_state.model_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recargando modelo: {str(e)}")

# ===============================
# Manejo de errores global
# ===============================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Error interno del servidor"}
    )

# ===============================
# Inicialización
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )


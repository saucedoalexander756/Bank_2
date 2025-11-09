import requests
import mysql.connector
import time
from datetime import datetime
############################CONTRASEÑAMYSQL

# ========================
# CONFIGURACIÓN
# ========================
API_URL = "http://127.0.0.1:8000/metrics"  # Usamos 127.0.0.1 para consistencia
DB_HOST = "localhost"
DB_PORT = 3306            
DB_USER = "root"
DB_PASSWORD = "root1234"
DB_NAME = "monitor_modelos"
INTERVAL_SECONDS = 3600    # Frecuencia: 3600 segundos = 1 hora

def connect_db():
    """Establece la conexión a la base de datos MySQL."""
    print("Intentando conectar a MySQL...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Conexión a MySQL exitosa.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ ERROR al conectar a MySQL: {err}")
        return None

def fetch_metrics():
    """Consulta el endpoint /metrics de la API y maneja errores HTTP 500."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Consultando API en: {API_URL}")
    try:
        response = requests.get(API_URL, timeout=10)
        
        if response.status_code == 500:
            error_data = response.json()
            print(f"❌ ERROR de la API (500): Las métricas no están disponibles.")
            print(f"   Detalles del Error de ML: {error_data.get('details', 'No hay detalles de error.')}")
            return None

        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ ERROR DE CONEXIÓN: La API no responde en {API_URL}.")
        print(f"   Verifique que la API esté activa en otra terminal. ({e})")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al consultar la API: {e}")
        return None

def save_metrics(metrics: dict, conn):
    """Guarda las métricas en la tabla metricas_rendimiento."""
    if not conn or not metrics:
        print("No se puede guardar, conexión o métricas faltantes.")
        return

    cursor = conn.cursor()
    
    data = (
        metrics.get("Accuracy"),
        metrics.get("Precision"),
        metrics.get("Recall"),
        metrics.get("F1_Score"),
        metrics.get("ROC_AUC"),
        metrics.get("tn"),
        metrics.get("fp"),
        metrics.get("fn"),
        metrics.get("tp"),
        metrics.get("Modelo")
    )

    sql = """
    INSERT INTO metricas_rendimiento (
        accuracy, precision_score, recall_score, f1_score, roc_auc,
        tn, fp, fn, tp, model_name
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        cursor.execute(sql, data)
        conn.commit()
        print("✅ Métricas guardadas exitosamente en la base de datos.")
    except mysql.connector.Error as err:
        print(f"❌ ERROR al guardar métricas en DB: {err}")
        conn.rollback()
    finally:
        cursor.close()

def main():
    """Bucle principal de monitoreo."""
    db_conn = connect_db()
    if not db_conn:
        print("El script de monitoreo se detiene debido al error de conexión a la DB.")
        return

    while True:
        metrics = fetch_metrics()
        if metrics:
            save_metrics(metrics, db_conn)
        
        print(f"Esperando {INTERVAL_SECONDS/60} minutos para la próxima ejecución...")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()


import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import requests 
import numpy as np
import os 
from contextlib import contextmanager
from io import BytesIO
import plotly.express as px
import scipy.stats as stats # NUEVA DEPENDENCIA: Para la prueba de hipótesis real

# --- 🛠️ Configuración de la página y Estilos (Se mantienen) ---
st.set_page_config(
    page_title="Support Vector Machine Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paleta de Colores y Fuente Formal: (Estilos CSS existentes)
st.markdown("""
<style>
/* ... (Estilos CSS existentes) ... */
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}
div.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
body {background-color: #f8f9fa;}
h1, h2, h3 {color: #212529; font-weight: 600;}
.metric-card {
    background-color: #ffffff; 
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 10px;
    border-left: 5px solid;
    transition: all 0.3s ease-in-out;
    width: 100%;
}
.metric-card:hover {
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
}
.metric-value {
    font-size: 2.2em;
    font-weight: bold; 
    margin-top: 5px;
    margin-bottom: 5px;
}
.metric-accuracy { border-color: #007bff; }
.metric-precision { border-color: #28a745; }
.metric-recall { border-color: #ffc107; }
.metric-f1_score { border-color: #6f42c1; }
.value-accuracy { color: #007bff; }
.value-precision { color: #28a745; }
.value-recall { color: #ffc107; }
.value-f1_score { color: #6f42c1; }
.data-container {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    box-shadow: none;
    margin-bottom: 20px;
}
.cm-header {font-weight: bold; text-align: center; color: #495057; border-bottom: 2px solid #adb5bd;}
.cm-cell {
    border: 1px solid #e9ecef; 
    padding: 12px; 
    text-align: center; 
    font-size: 1.2em; 
    font-weight: 500; 
    border-radius: 0px; 
    vertical-align: middle;
}
.cm-tn { background-color: #f1f3f5; color: #495057; } 
.cm-fp { background-color: #fae3e6; color: #dc3545; } 
.cm-fn { background-color: #fff3cd; color: #fd7e14; } 
.cm-tp { background-color: #d4edda; color: #28a745; } 
.cm-label {font-weight: bold; padding-right: 10px; color: #212529; text-align: right;}
.stDataFrame {
    border: 1px solid #e9ecef;
    border-radius: 4px;
}
.metric-divider {
    border-top: 1px dashed #ced4da; 
    margin: 10px 0;
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# Al inicio del archivo, después de los imports:
import os

# Configuración para producción/desarrollo
if os.environ.get("RAILWAY_ENVIRONMENT"):
    # En Railway
    API_BASE_URL = os.environ.get("API_URL", "")  # URL de tu API en Railway
else:
    # Desarrollo local
    API_BASE_URL = "http://127.0.0.1:8000"

# El resto del código permanece igual...

# Datos de simulación (se mantienen como respaldo si la API falla)
data_simulacion = {
    "metrics": {
        "precision": 0.952, "recall": 0.909, "accuracy": 0.925, "f1_score": 0.930, "roc_auc": 0.94,
    },
    "confusionMatrix": [[850, 50], [100, 1000]],
    "parameters": {
        "Modelo": "SVM (Modo Estático / Fallback)",
        "Kernel": "rbf",
        "C": 1.0,
        "Gamma": "scale"
    },
}

@contextmanager
def suppress_streamlit_output():
    # ... (función para suprimir output de Streamlit, se mantiene)
    st_info_orig = st.info
    st_warning_orig = st.warning
    st_error_orig = st.error
    st_success_orig = st.success

    def no_op(*args, **kwargs):
        pass
    
    st.info = no_op
    st.warning = no_op
    st.error = no_op
    st.success = st_success_orig 

    try:
        yield
    finally:
        st.info = st_info_orig
        st.warning = st_warning_orig
        st.error = st_error_orig
        st.success = st_success_orig


@st.cache_data(ttl=60) # Cache por 60 segundos
def fetch_metrics_from_api(base_url):
    """Obtiene las métricas más recientes del endpoint /metrics."""
    metrics_url = f"{base_url}/metrics"
    try:
        response = requests.get(metrics_url, timeout=5) 
        if response.status_code == 200:
            data = response.json()
            
            # Normalización y renombramiento de claves para consistencia
            data['metrics'] = {
                'accuracy': data.pop('Accuracy'),
                'precision': data.pop('Precision'),
                'recall': data.pop('Recall'),
                'f1_score': data.pop('F1_Score'),
                'roc_auc': data.pop('ROC_AUC')

            }
            # La CM viene como una lista de listas [[TN, FP], [FN, TP]]
            data['confusionMatrix'] = data.pop('Confusion_Matrix')
            data['parameters'] = {
                'Modelo': data.pop('Modelo', 'SVM (API)'),
                'Kernel': 'N/A', 'C': 'N/A', 'Gamma': 'N/A' # Parámetros faltantes en este endpoint
            }
            return data
        else:
            st.error(f"⚠️ API /metrics devolvió error {response.status_code}. Usando fallback.")
            return data_simulacion
            
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error al conectar con la API ({metrics_url}). Usando datos de simulación.")
        return data_simulacion
    

@st.cache_data(ttl=60) # Cache por 60 segundos
def fetch_history_from_api(base_url):
    """
    NUEVA FUNCIÓN REAL: Obtiene los datos históricos del endpoint /history.
    """
    history_url = f"{base_url}/history"
    try:
        response = requests.get(history_url, timeout=10)
        if response.status_code == 200:
            data_list = response.json()
            if isinstance(data_list, list) and data_list:
                history_df = pd.DataFrame(data_list)
                # Renombrar 'timestamp' a 'Timestamp' para consistencia con Plotly
                history_df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
                history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
                # Renombrar columnas para display (ej: accuracy -> Accuracy)
                history_df.columns = [col.capitalize() if col != 'Timestamp' else col for col in history_df.columns]
                return history_df
            elif data_list and data_list.get("message") == "No hay registros históricos disponibles.":
                 st.info("La API no tiene registros históricos en la base de datos.")
                 return None
            else:
                 st.warning("El endpoint /history devolvió un formato inesperado.")
                 return None
        else:
            st.error(f"API /history devolvió error {response.status_code}.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar al endpoint histórico: {history_url}. ({e})")
        return None


@st.cache_data
def fetch_plot_from_api(base_url, endpoint):
    """Obtiene una imagen de gráfica de un endpoint."""
    plot_url = f"{base_url}{endpoint}"
    try:
        response = requests.get(plot_url, timeout=10) 
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return response.content
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudo cargar la gráfica desde el endpoint: {plot_url}")
        return None
    return None

# ----------------------------------------------------
# FUNCIÓN REAL: Prueba de Hipótesis con SciPy (Punto 4)
# ----------------------------------------------------
def hypothesis_test_scipy(target_mean, historical_data):
    """Ejecuta una Prueba t de una muestra para la media de Accuracy usando SciPy."""
        
    accuracy_data = historical_data['Accuracy']
    
    if len(accuracy_data) < 2:
        return {
            "H0": f"La Accuracy media es $\\mu = {target_mean}$",
            "Ha": f"La Accuracy media es $\\mu \\ne {target_mean}$",
            "Muestra_Media": accuracy_data.mean() if len(accuracy_data) > 0 else 0,
            "Estadistico_t": 0.0,
            "P_Valor": 1.0, 
            "Alpha": 0.05,
            "Decision": "No hay suficientes datos (N<2) para realizar la Prueba t.",
            "Result_Type": 'warning',
        }

    # Ejecutar la prueba t de una muestra: (data, media_hipotética, alternativa)
    t_statistic, p_value = stats.ttest_1samp(
        a=accuracy_data, 
        popmean=target_mean, 
        alternative='two-sided' # Prueba bilateral (es diferente)
    )
    
    sample_mean = accuracy_data.mean()
    alpha = 0.05
    
    # Decisión
    if p_value < alpha:
        decision = f"Rechazar $H_0$. Hay evidencia estadísticamente significativa ($\alpha = {alpha}$) de que la **Accuracy media ({sample_mean:.4f}) es diferente** del valor objetivo ({target_mean:.2f})."
        result_type = 'error' # Usamos error para indicar un resultado significativo (de cambio)
    else:
        decision = f"No rechazar $H_0$. No hay evidencia estadísticamente significativa ($\alpha = {alpha}$) de que la Accuracy media ({sample_mean:.4f}) sea diferente del valor objetivo ({target_mean:.2f})."
        result_type = 'success' # Usamos success para indicar que es consistente con el objetivo
        
    return {
        "H0": f"La Accuracy media del modelo es igual al valor objetivo: $\\mu = {target_mean}$",
        "Ha": f"La Accuracy media del modelo es diferente del valor objetivo: $\\mu \\ne {target_mean}$",
        "Muestra_Media": sample_mean,
        "Estadistico_t": t_statistic,
        "P_Valor": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Result_Type": result_type,
    }


# --- Lógica Principal del Dashboard ---

with suppress_streamlit_output():
    # Obtención de datos más recientes
    data = fetch_metrics_from_api(API_BASE_URL)
    # Obtención de datos históricos (REAL)
    history_df = fetch_history_from_api(API_BASE_URL)


if data:
    # --- Encabezado y Pestañas ---
    st.markdown("<h2>Support Vector Machine Dashboard</h2>", unsafe_allow_html=True) 
    
    params = data.get('parameters', data_simulacion['parameters']) 
    st.sidebar.title("🛠️ Configuración del Modelo")
    st.sidebar.markdown(f"**Modelo:** **`{params['Modelo']}`**")
    st.sidebar.markdown(f"**Kernel:** **`{params.get('Kernel', 'N/A')}`**")

    # NUEVAS PESTAÑAS
    tab_metrics, tab_curves, tab_history, tab_hypothesis, tab_details = st.tabs([
        "Métricas Principales", 
        "Curvas de Rendimiento", 
        "Histórico de Métricas", 
        "Prueba de Hipótesis",
        "Detalles del JSON"
    ])

    # ... (Contenido de tab_metrics y tab_curves se mantiene igual) ...
    with tab_metrics:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.header("Métricas de Rendimiento General")

            metrics = data['metrics']
            
            def format_metric(value, color_class):
                # Manejar el caso de ROC_AUC que no está en el rango 0-1
                if color_class == 'value-accuracy' and 'ROC_AUC' in color_class:
                    return f'<div class="metric-value {color_class}">{value:.3f}</div>'
                return f'<div class="metric-value {color_class}">{value*100:.1f}%</div>'

            # Métricas
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card metric-precision">Precision', unsafe_allow_html=True)
                st.markdown(format_metric(metrics['precision'], 'value-precision'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card metric-accuracy">Accuracy (Exactitud)', unsafe_allow_html=True)
                st.markdown(format_metric(metrics['accuracy'], 'value-accuracy'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card metric-accuracy">ROC-AUC', unsafe_allow_html=True)
                st.markdown(format_metric(metrics['roc_auc'], 'value-accuracy'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card metric-recall">Recall (Sensibilidad)', unsafe_allow_html=True)
                st.markdown(format_metric(metrics['recall'], 'value-recall'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card metric-f1_score">F1-Score', unsafe_allow_html=True)
                st.markdown(format_metric(metrics['f1_score'], 'value-f1_score'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


        with col_right:
            # Matriz de Confusión
            st.subheader("Matriz de Confusión")
            cm = data['confusionMatrix']
            TN, FP = cm[0]
            FN, TP = cm[1]

            st.markdown(f"""
            <div class="data-container">
                <table style="width:100%; border-collapse: separate; border-spacing: 0px;">
                    <thead>
                        <tr><td></td><td colspan="2" class="cm-header" style="border-right: none;">Predicción:</td></tr>
                        <tr><td></td><td class="cm-header">Clase No</td><td class="cm-header">Clase Sí</td></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="cm-label" style="border-right: 2px solid #adb5bd;">Real: Clase No</td>
                            <td class="cm-cell cm-tn">{TN}<br><span style="font-size:0.7em;">Verdadero Negativo (TN)</span></td>
                            <td class="cm-cell cm-fp">{FP}<br><span style="font-size:0.7em;">Falso Positivo (FP)</span></td>
                        </tr>
                        <tr>
                            <td class="cm-label" style="border-right: 2px solid #adb5bd;">Real: Clase Sí</td>
                            <td class="cm-cell cm-fn">{FN}<br><span style="font-size:0.7em;">Falso Negativo (FN)</span></td>
                            <td class="cm-cell cm-tp">{TP}<br><span style="font-size:0.7em;">Verdadero Positivo (TP)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # Parámetros del Modelo
            st.subheader("Parámetros del Modelo SVM")
            params_df = pd.DataFrame(list(params.items()), columns=['Parámetro', 'Valor'])
            params_df.set_index('Parámetro', inplace=True)
            st.markdown('<div class="data-container">', unsafe_allow_html=True)
            st.dataframe(params_df, hide_index=False, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


    with tab_curves:
        st.header("Análisis Gráfico de Rendimiento")
        col_roc, col_pr = st.columns(2)
        
        with col_roc:
            st.markdown("### Curva ROC (Receiver Operating Characteristic)")
            roc_image_bytes = fetch_plot_from_api(API_BASE_URL, "/plot/roc")
            if roc_image_bytes:
                st.image(BytesIO(roc_image_bytes), caption="Curva ROC generada por la API")
            else:
                st.warning("No se pudo obtener la gráfica ROC de la API.")

        with col_pr:
            st.markdown("### Curva Precision-Recall")
            pr_image_bytes = fetch_plot_from_api(API_BASE_URL, "/plot/precision_recall")
            if pr_image_bytes:
                st.image(BytesIO(pr_image_bytes), caption="Curva Precision-Recall generada por la API")
            else:
                st.warning("No se pudo obtener la gráfica Precision-Recall de la API.")
                
    # ----------------------------------------------------
    # CONTENIDO NUEVO: Histórico de Métricas (Punto 3 - REAL)
    # ----------------------------------------------------
    with tab_history:
        st.header("Historico de Métricas del Modelo a lo largo del Tiempo")
        
        if history_df is not None and not history_df.empty:
            st.write(f"Visualización de las tendencias para **{len(history_df)}** registros.")
    
            metric_to_plot = st.selectbox(
                "Selecciona la Métrica a Visualizar:",
                ('Accuracy', 'Recall', 'F1_score', 'Precision', 'Roc_auc'),
                index=0
            )
            
            # Gráfico interactivo con Plotly
            fig = px.line(
                history_df, 
                x='Timestamp', 
                y=metric_to_plot, 
                title=f'Tendencia Histórica de {metric_to_plot}',
                labels={'Timestamp': 'Fecha/Hora', metric_to_plot: metric_to_plot},
                markers=True
            )
            
            fig.update_layout(
                plot_bgcolor='#f8f9fa', 
                paper_bgcolor='#ffffff', 
                xaxis_title="Fecha", 
                yaxis_title=metric_to_plot,
                yaxis_tickformat='.3f',
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Datos Históricos (Primeros y Últimos Registros)")
            st.dataframe(history_df.tail(10).sort_values('Timestamp', ascending=False), use_container_width=True)
        else:
            st.warning("No se pudo cargar el historial de métricas de la API o la base de datos está vacía. Asegúrate de que la API esté corriendo.")


    # ----------------------------------------------------
    # CONTENIDO NUEVO: Prueba de Hipótesis (Punto 4 - REAL con SciPy)
    # ----------------------------------------------------
    with tab_hypothesis:
        st.header("Análisis de Prueba de Hipótesis (Prueba t de una muestra)")
        
        if history_df is not None and len(history_df) >= 2:
            st.write("Verificación de si la **Accuracy media** del modelo es significativamente diferente de un valor objetivo ($\mu$).")
            st.markdown('<hr style="border-top: 2px solid #e9ecef;">', unsafe_allow_html=True)
            
            col_params, col_results = st.columns([1, 2])
            
            with col_params:
                st.subheader("Configuración de la Prueba")
                
                # Parámetro de entrada (media objetivo)
                target_accuracy = st.number_input(
                    'Valor Objetivo de la Accuracy (µ)',
                    min_value=0.5,
                    max_value=1.0,
                    value=0.90,
                    step=0.01,
                    format="%.2f",
                    key="target_acc"
                )
                
                st.markdown(f"**Nivel de Significación ($\alpha$):** **`0.05`**")
                st.markdown(f"**Muestra ($N$):** **`Accuracy`** de los {len(history_df)} registros históricos.")
                
                # Ejecutar la prueba
                results = hypothesis_test_scipy(target_accuracy, history_df)
                
            with col_results:
                st.subheader("Resultados de la Prueba T")
                
                # Formulación de Hipótesis
                st.markdown('**Hipótesis Nula ($H_0$):**', unsafe_allow_html=True)
                st.code(results['H0'], language='markdown')
                st.markdown('**Hipótesis Alternativa ($H_a$):**', unsafe_allow_html=True)
                st.code(results['Ha'], language='markdown')
                
                st.markdown('<hr style="border-top: 1px dashed #ced4da;">', unsafe_allow_html=True)
                
                col_t, col_p = st.columns(2)
                with col_t:
                    st.metric("Estadístico t", f"{results['Estadistico_t']:.3f}", delta_color="off")
                with col_p:
                    st.metric("p-valor", f"{results['P_Valor']:.4f}", delta_color="off")
                    
                st.markdown('<hr style="border-top: 1px dashed #ced4da;">', unsafe_allow_html=True)
                
                # Interpretación y Decisión
                if results['Result_Type'] == 'success':
                    st.success(f"**DECISIÓN:** {results['Decision']}")
                elif results['Result_Type'] == 'error':
                    st.error(f"**DECISIÓN:** {results['Decision']}")
                else:
                    st.warning(f"**DECISIÓN:** {results['Decision']}")
                
                st.caption(f"Media de la Muestra: {results['Muestra_Media']:.4f}")
        else:
            st.warning("Se requieren al menos 2 registros históricos para realizar la Prueba t de Hipótesis.")


    with tab_details:
        st.header("Contenido Completo del JSON de Métricas Recientes")
        st.json(data)
        st.header("Contenido de Datos Históricos (API /history)")
        if history_df is not None:
             st.dataframe(history_df, use_container_width=True)
        else:
             st.info("No hay datos históricos para mostrar.")

# Si la conexión falla y no hay fallback
if not data:
    st.error("No se pudo obtener ninguna métrica. Asegúrate de que la API de FastAPI esté corriendo en `http://127.0.0.1:8000`.")

    
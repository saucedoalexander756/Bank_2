-- 1. CREAR LA BASE DE DATOS (si no existe)
CREATE DATABASE IF NOT EXISTS monitor_modelos;

-- 2. SELECCIONAR LA BASE DE DATOS
USE monitor_modelos;

-- 3. CREAR LA TABLA DE MÉTRICAS DE RENDIMIENTO
CREATE TABLE metricas_rendimiento (
    -- Clave Primaria
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- Marca de Tiempo (esencial para el monitoreo)
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Información del modelo
    model_name VARCHAR(50) NOT NULL DEFAULT 'SVM',
    
    -- Métricas de rendimiento
    accuracy DECIMAL(6, 4) NOT NULL,
    precision_score DECIMAL(6, 4) NOT NULL,
    recall_score DECIMAL(6, 4) NOT NULL,
    f1_score DECIMAL(6, 4) NOT NULL,
    roc_auc DECIMAL(6, 4) NOT NULL,
    
    -- Componentes de la Matriz de Confusión (para análisis detallado)
    tn INT NOT NULL,  -- True Negatives
    fp INT NOT NULL,  -- False Positives
    fn INT NOT NULL,  -- False Negatives
    tp INT NOT NULL   -- True Positives
);

-- Opcional: Mostrar la estructura de la tabla para verificar
-- DESCRIBE metricas_rendimiento;
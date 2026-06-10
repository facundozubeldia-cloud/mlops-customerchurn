# Proyecto MLOps — AndesLink: Predicción de Churn

**Alumnos:** Facundo Zubeldia · Gonzalo Martín González Nastovich · Agustín Meza  
**Materia:** Laboratorio de Minería de Datos — ISTEA  
**Profesor:** Diego Mosquera

---

## Descripción

Este repositorio contiene el desarrollo integral de una solución de Machine Learning bajo prácticas de MLOps para la empresa ficticia **AndesLink Servicios Digitales S.A.**, que necesita anticipar el abandono de clientes (churn) para activar campañas de retención.

El proyecto cubre el ciclo completo: entrenamiento, despliegue y monitoreo.

---

## Estructura del proyecto

```
Lab-Mineria-de-Datos/
├── data/
│   └── churn_sintetico.csv.dvc      # dataset versionado con DVC
├── models/
│   └── model_churn.pkl              # modelo serializado listo para inferencia
├── notebooks/
│   └── 01_eda_customer_churn.ipynb  # EDA, preprocesamiento y entrenamiento
├── src/
│   ├── api.py                       # API de inferencia FastAPI
│   ├── app.py                       # dashboard Streamlit
│   └── train.py                     # script de entrenamiento reutilizable
├── reports/
│   ├── INFORME_TECNICO_ANDESLINK_PARCIAL_1.pdf
│   └── test_report.html             # reporte de tests pytest
├── scripts/
│   └── 02_verificación_modelo_churn.py
├── tests/
│   └── test_api.py                  # tests pytest de la API
├── conftest.py                      # configuración pytest
├── Dockerfile                       # imagen Docker para la API
├── Dockerfile.streamlit             # imagen Docker para el dashboard
├── docker-compose.yml               # orquestación de servicios
├── requirements.txt                 # dependencias Python (entorno local)
├── requirements-docker.txt          # dependencias mínimas para Docker
├── environment.yml                  # entorno Conda reproducible
└── README.md
```

---

## Stack tecnológico

| Herramienta | Uso | Entrega |
|---|---|---|
| Python 3.11 | Lenguaje base | Todas |
| pandas / numpy | Procesamiento de datos | Todas |
| scikit-learn | Entrenamiento y evaluación | Entrega 1 |
| MLflow | Tracking de experimentos | Entrega 1 |
| DVC | Versionado del dataset | Entrega 1 |
| Git | Versionado del código | Todas |
| FastAPI | API de inferencia | Entrega 2 |
| Streamlit | Dashboard de predicción | Entrega 2 |
| Docker + Compose | Contenedorización | Entrega 2 |
| pytest | Tests automatizados | Entrega 2 |
| Prometheus + Grafana | Monitoreo técnico | Entrega 3 |
| Evidently | Monitoreo de drift | Entrega 3 |

---

## Instalación y configuración del entorno

### Requisitos previos
- Python 3.11
- Docker y Docker Compose
- Git
- DVC (`pip install dvc`)

### 1. Clonar el repositorio

```bash
git clone https://github.com/facundozubeldia-cloud/Lab-Mineria-de-Datos.git
cd Lab-Mineria-de-Datos
```

### 2. Crear el entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

O con Conda:

```bash
conda env create -f environment.yml
conda activate lab-mineria
```

### 3. Bajar el dataset con DVC

```bash
dvc pull
```

---

## Despliegue con Docker

Levantá los dos servicios con un solo comando desde la raíz del proyecto:

```bash
docker-compose up --build -d
```

Esto inicia:
- **API de inferencia** en `http://localhost:8000`
- **Dashboard Streamlit** en `http://localhost:8501`

Para verificar que los servicios están corriendo:

```bash
docker-compose ps
```

Para detenerlos:

```bash
docker-compose down
```

### Servicios disponibles

| Servicio | URL | Descripción |
|---|---|---|
| API FastAPI | `http://localhost:8000/docs` | Documentación interactiva Swagger |
| API predict | `POST http://localhost:8000/predict` | Endpoint de inferencia |
| Dashboard | `http://localhost:8501` | Interfaz visual para predicción |

---

## Tests

Correr los tests desde la raíz del proyecto:

```bash
pytest tests/ -v
```

Para generar reporte HTML:

```bash
pytest tests/ -v --html=reports/test_report.html --self-contained-html
```

Resultados actuales: **7/7 tests pasados**. El reporte se guarda en `reports/test_report.html`.

---

## Ejecución sin Docker

### Opción A — Notebook completo

```bash
cd notebooks/
jupyter lab
```

Abrí `01_eda_customer_churn.ipynb` y ejecutá **Kernel → Restart & Run All**.

### Opción B — Script de entrenamiento

```bash
python src/train.py
```

### Opción C — Servicios locales (sin Docker)

Terminal 1 — API:
```bash
uvicorn src.api:app --reload --port 8000
```

Terminal 2 — Dashboard:
```bash
streamlit run src/app.py
```

---

## Tracking de experimentos con MLflow

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5001
```

Abrí `localhost:5001` → pestaña **Model training** → experimento `AndesLink_Churn_Prediction`.

---

## Estrategia de reproducibilidad

- **Git** versiona el código fuente y archivos de configuración.
- **DVC** está configurado para versionado del dataset (`churn_sintetico.csv.dvc`). 
  El archivo `.dvc` se mantiene para trazabilidad. El CSV se incluye directamente 
  en el repositorio dado que es un dataset sintético de tamaño reducido (5.000 registros). 
  En un entorno productivo el remote apuntaría a S3 o GCS con autenticación por 
  Service Account.
- **MLflow** registra todos los experimentos, hiperparámetros y métricas.
- **Docker** garantiza reproducibilidad del despliegue.

---

## Resultados del modelo (con split estratificado)

| Modelo | Accuracy | F1 (churn) | Recall (churn) | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.672 | **0.602** | **0.729** | **0.758** |
| Decision Tree | 0.635 | 0.570 | 0.712 | 0.708 |
| Random Forest | 0.707 | 0.483 | 0.403 | 0.726 |

**Modelo seleccionado: Logistic Regression** — mayor F1-score y Recall en la clase churn. El split estratificado garantiza que el desbalance 66/34 se mantiene consistente entre train y test.
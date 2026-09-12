# Proyecto MLOps — AndesLink: Predicción de Churn

# Autoria 

-Facundo Zubeldia 

---

## Descripción

Este repositorio contiene el desarrollo integral de una solución de Machine Learning bajo prácticas de MLOps para la empresa ficticia **AndesLink Servicios Digitales S.A.**, que necesita anticipar el abandono de clientes (churn) para activar campañas de retención.

El proyecto cubre el ciclo completo de MLOps: entrenamiento, despliegue y monitoreo.

---
## Documentacíon

La totalidad de la documentación que abarca las correcciones de la entrega 1 y 2, como tambien la entrega 3, se encuentran en el archivo /reports/DOCUMENTACIÓN_FINAL.DOCX

Asimismo se encuentra la ppt utilizada en el video de presentación, como tambien las entregas 1 y 2.

## ⚠️ Recomendación de uso

**Para correr el proyecto, usá Docker** — es el camino recomendado y validado por el equipo:

```bash
docker-compose up --build -d
```

# Comentarios

En el caso de que utilices multiples predicciones en el modelo de andeslink, por favor, eliminar cache con ctlr + f5 (esto resetea cache y permite que la pagina no se resetee con cada predicción).

El entorno local con `requirements.txt` está pensado para desarrollo (notebook, MLflow, DVC) y puede requerir ajustes según el sistema operativo. Para correr únicamente la API y el dashboard sin Docker, usá `requirements-docker.txt`.


Asimismo, en caso de que algun puerto este ocupado por multiples ejecuciones:
docker-compose down -v
docker-compose up --build -d
esto tira
---

## Estructura del proyecto

```
Lab-Mineria-de-Datos/
├── data/
│   ├── churn_sintetico.csv          # dataset incluido en el repo
│   └── churn_sintetico.csv.dvc      # metadata de versionado DVC
├── models/
│   └── model_churn.pkl              # modelo serializado listo para inferencia
├── monitoring/
│   └── prometheus.yml               # configuración de scraping de Prometheus
│   └── grafana_dashboard.json
│   └── provisioning
│       └──Dashboards
│          └──dashboards.yml 
│       └──Datasources
│          └──datasources.yml          
├── notebooks/
│   ├── 01_eda_customer_churn.ipynb  # EDA, preprocesamiento y entrenamiento
│   └── mlflow.db                    # tracking de experimentos (SQLite)
├── src/
│   ├── api.py                       # API de inferencia FastAPI + métricas Prometheus
│   ├── app.py                       # dashboard Streamlit
│   └── train.py                     # script de entrenamiento reutilizable
├── reports/
│   ├── INFORME_TECNICO_ANDESLINK_PARCIAL_1.pdf
│   ├── INFORME_TECNICO_ANDESLINK_PARCIAL_2.pdf
│   ├── DOCUMENTACIÓN_FINAL.docx
│   ├── reporte_evidently.html       # reporte de drift de datos
│   └── test_report.html             # reporte de tests pytest
├── scripts/
│   ├── 02_verificación_modelo_churn.py
│   └── generar_reporte_evidently.py # script de monitoreo de drift
├── tests/
│   └── test_api.py                  # 11 tests pytest de la API
├── .github/workflows/
│   ├── ci.yml                       # CI: corre los tests con cada push
│   └── docker-build.yml             # CI: valida que las imágenes Docker buildean
├── conftest.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml               # orquesta API, dashboard, Prometheus y Grafana
├── requirements.txt                 # dependencias completas (desarrollo local)
├── requirements-docker.txt          # dependencias mínimas (producción)
├── environment.yml
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
| Git + GitHub Actions | Versionado y CI/CD | Todas |
| FastAPI | API de inferencia | Entrega 2 |
| Streamlit | Dashboard de predicción | Entrega 2 |
| Docker + Compose | Contenedorización | Entrega 2 |
| pytest | Tests automatizados (11 tests) | Entrega 2 |
| Prometheus | Recolección de métricas técnicas | Entrega 3 |
| Grafana | Visualización de métricas | Entrega 3 |
| Evidently | Monitoreo de drift de datos | Entrega 3 |

---

## Despliegue con Docker (recomendado)

### Requisitos previos
- Docker y Docker Compose instalados
- Git

### Pasos

```bash
git clone https://github.com/facundozubeldia-cloud/Lab-Mineria-de-Datos.git
cd Lab-Mineria-de-Datos
docker-compose up --build -d
```

Esto levanta 4 servicios:

| Servicio | URL | Descripción |
|---|---|---|
| API FastAPI | `http://localhost:8000/docs` | Documentación interactiva Swagger |
| API predict | `POST http://localhost:8000/predict` | Endpoint de inferencia |
| API health | `GET http://localhost:8000/health` | Healthcheck con validación del modelo |
| API metrics | `GET http://localhost:8000/metrics` | Métricas para Prometheus |
| Dashboard | `http://localhost:8501` | Interfaz visual para predicción |
| Prometheus | `http://localhost:9091` | Base de datos de métricas |
| Grafana | `http://localhost:3001` | Dashboard de monitoreo (admin/admin) |

### Comandos útiles

```bash
docker-compose ps          # ver estado de los servicios
docker-compose logs -f     # ver logs en tiempo real
docker-compose down        # detener y eliminar los contenedores
```

```
docker-compose down -v  
docker-compose up --build -d   # detiene los contenedores y vuelve a correrlos.
```
---

## Monitoreo

### Prometheus + Grafana

Prometheus recolecta métricas de la API cada 15 segundos desde `/metrics`. El dashboard de Grafana muestra:

- **Tráfico por endpoint** — cantidad de requests a `/health`, `/predict` y `/metrics`
- **Status codes** — distribución de respuestas 2xx, 4xx, 5xx
- **Latencia del /predict** — tiempo promedio de inferencia en producción
- **Uptime** — tiempo de actividad del servicio

### Evidently — Monitoreo de drift

Para generar el reporte de calidad de datos y drift:

```bash
python scripts/generar_reporte_evidently.py
```

Esto genera `reports/reporte_evidently.html` comparando el dataset de entrenamiento (referencia) contra datos de producción simulados. Abrís el archivo en el navegador para ver el análisis completo.

En el último reporte se detectó drift en **10 de 15 columnas (66.7%)**, con las variables más afectadas siendo `support_tickets`, `monthly_charge` y `tenure_months` — señal de que el perfil de clientes en producción difiere del dataset de entrenamiento.

---

## Tests

```bash
pytest tests/ -v
```

Con reporte HTML:

```bash
pytest tests/ -v --html=reports/test_report.html --self-contained-html
```

**Resultado actual: 11/11 tests pasados.** Los tests cubren: health check, predicción válida, validación de campos faltantes, tipos incorrectos, categorías inválidas (Literal), formato de respuesta, y casos de negocio extremos.

---

## Entorno de desarrollo local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Notebook completo (EDA + entrenamiento)

```bash
cd notebooks/
jupyter lab
```

### Script de entrenamiento

```bash
python src/train.py
```

### Tracking de experimentos con MLflow

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5001
```

Abrí `localhost:5001` → pestaña **Model training** → experimento `AndesLink_Churn_Prediction`.

---

## Estrategia de reproducibilidad

- **Git** versiona el código fuente y archivos de configuración.
- **DVC** está configurado para versionado del dataset. El CSV se incluye además directamente en el repositorio para facilitar la reproducción sin configuración adicional. En producción el remote apuntaría a S3 o GCS.
- **MLflow** registra todos los experimentos, hiperparámetros y métricas.
- **Docker** garantiza que todos los servicios corren de manera idéntica en cualquier máquina.
- **GitHub Actions** valida automáticamente con cada push que los tests pasen y que las imágenes Docker se construyan correctamente.

---

## Resultados del modelo

| Modelo | Accuracy | F1 (churn) | Recall (churn) | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.672 | **0.602** | **0.729** | **0.758** |
| Decision Tree | 0.635 | 0.570 | 0.712 | 0.708 |
| Random Forest | 0.707 | 0.483 | 0.403 | 0.726 |

**Modelo seleccionado: Logistic Regression** con `class_weight='balanced'` y split estratificado — mayor F1-score y Recall en la clase churn, que es la métrica relevante para el objetivo de negocio de AndesLink.

---

## Documentación adicional

- `reports/INFORME_TECNICO_ANDESLINK_PARCIAL_1.pdf` — entrenamiento: EDA, preparación de datos, comparación de modelos.
- `reports/INFORME_TECNICO_ANDESLINK_PARCIAL_2.pdf` — despliegue: arquitectura, API, Docker, tests.
- `reports/INFORME_TECNICO_ANDESLINK_PARCIAL_3.docx` — monitoreo: Prometheus, Grafana, Evidently, análisis de drift.
- `reports/reporte_evidently.html` — reporte interactivo de drift de datos.
- `reports/test_report.html` — reporte detallado de los 11 tests automatizados.
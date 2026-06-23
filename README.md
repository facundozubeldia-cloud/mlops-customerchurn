# Proyecto MLOps — AndesLink: Predicción de Churn

**Alumnos:** Facundo Zubeldia · Gonzalo Martín González Nastovich · Agustín Meza  
**Materia:** Laboratorio de Minería de Datos — ISTEA  
**Profesor:** Ing. Diego Mosquera

---

## Descripción

Este repositorio contiene el desarrollo integral de una solución de Machine Learning bajo prácticas de MLOps para la empresa ficticia **AndesLink Servicios Digitales S.A.**, que necesita anticipar el abandono de clientes (churn) para activar campañas de retención.

El proyecto cubre el ciclo completo: entrenamiento, despliegue y monitoreo.

---

## ⚠️ Recomendación de uso

**Para correr el proyecto, usá Docker** — es el camino probado, validado por todo el equipo y sin conflictos de dependencias:

```bash
docker-compose up --build -d
```

La instalación local con `requirements.txt` (necesaria solo para el notebook, MLflow UI o DVC) puede presentar conflictos de versiones entre paquetes según el sistema operativo, dado que combina librerías de tracking, despliegue y desarrollo en un mismo archivo. Si necesitás un entorno local liviano y sin conflictos, usá `requirements-docker.txt`, pensado específicamente para correr la API y el dashboard.

---

## Estructura del proyecto

```
Lab-Mineria-de-Datos/
├── data/
│   ├── churn_sintetico.csv          # dataset (incluido en el repo, ver nota DVC en estrategias de producibilidad)
│   └── churn_sintetico.csv.dvc      # metadata de versionado DVC
├── models/
│   └── model_churn.pkl              # modelo serializado listo para inferencia
├── notebooks/
│   ├── 01_eda_customer_churn.ipynb  # EDA, preprocesamiento y entrenamiento
│   └── mlflow.db                    # tracking de experimentos (SQLite)
├── src/
│   ├── api.py                       # API de inferencia FastAPI
│   ├── app.py                       # dashboard Streamlit
│   └── train.py                     # script de entrenamiento reutilizable
├── reports/
│   ├── INFORME_TECNICO_ANDESLINK_PARCIAL_1.pdf
│   ├── INFORME_TECNICO_ANDESLINK_PARCIAL_2.pdf
│   └── test_report.html             # reporte de tests pytest
├── scripts/
│   └── 02_verificación_modelo_churn.py
├── tests/
│   └── test_api.py                  # tests pytest de la API
├── .github/workflows/
│   ├── ci.yml                       # CI: corre los tests con cada push
│   └── docker-build.yml             # CI: valida que las imágenes Docker buildean
├── conftest.py                      # configuración pytest
├── Dockerfile                       # imagen Docker para la API
├── Dockerfile.streamlit             # imagen Docker para el dashboard
├── docker-compose.yml               # orquestación de servicios
├── requirements.txt                 # dependencias completas (desarrollo local)
├── requirements-docker.txt          # dependencias mínimas (Docker / producción)
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
| GitHub Actions | CI: tests y validación de build | Entrega 2 |
| Prometheus + Grafana | Monitoreo técnico | Entrega 3 |
| Evidently | Monitoreo de drift | Entrega 3 |

---

## Despliegue con Docker (recomendado)

### Requisitos previos
- Docker y Docker Compose
- Git

### Pasos

```bash
git clone https://github.com/facundozubeldia-cloud/Lab-Mineria-de-Datos.git
cd Lab-Mineria-de-Datos
docker-compose up --build -d
```

Esto inicia:
- **API de inferencia** en `http://localhost:8000`
- **Dashboard Streamlit** en `http://localhost:8501`

### Comandos útiles

```bash
docker-compose ps          # ver estado de los servicios
docker-compose logs -f     # ver logs en tiempo real
docker-compose down        # detener y eliminar los contenedores
```

### Servicios disponibles

| Servicio | URL | Descripción |
|---|---|---|
| API FastAPI | `http://localhost:8000/docs` | Documentación interactiva Swagger |
| API predict | `POST http://localhost:8000/predict` | Endpoint de inferencia |
| Dashboard | `http://localhost:8501` | Interfaz visual para predicción |

---

## Tests

Los tests corren tanto en local como automáticamente en GitHub Actions con cada push.

```bash
pytest tests/ -v
```

Con reporte HTML:

```bash
pytest tests/ -v --html=reports/test_report.html --self-contained-html
```

**Resultado actual: 7/7 tests pasados.** Reporte disponible en `reports/test_report.html`.

---

## Desarrollo local (notebook, entrenamiento, MLflow, DVC)

Para esta sección se recomienda crear un entorno virtual exclusivo, dado que `requirements.txt` incluye dependencias de desarrollo que pueden generar conflictos según el sistema operativo.

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Notebook completo (EDA + entrenamiento)

```bash
cd notebooks/
jupyter lab
```

Abrí `01_eda_customer_churn.ipynb` y ejecutá **Kernel → Restart & Run All**.

### Script de entrenamiento

```bash
python src/train.py
```

### Servicios sin Docker (alternativa)

Terminal 1 — API:
```bash
uvicorn src.api:app --reload --port 8000
```

Terminal 2 — Dashboard:
```bash
streamlit run src/app.py
```

### Tracking de experimentos con MLflow

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5001
```

Abrí `localhost:5001` → pestaña **Model training** → experimento `AndesLink_Churn_Prediction`.

---

## Estrategia de reproducibilidad

- **Git** versiona el código fuente y archivos de configuración.
- **DVC** está configurado para el versionado del dataset (`churn_sintetico.csv.dvc`). El archivo `.dvc` se mantiene para trazabilidad. El CSV se incluye además directamente en el repositorio, dado que es un dataset sintético de tamaño reducido (5.000 registros) y esto evita depender de un remote externo para reproducir el proyecto. En un entorno productivo, el remote apuntaría a S3 o GCS con autenticación por Service Account.
- **MLflow** registra todos los experimentos, hiperparámetros y métricas (`notebooks/mlflow.db`, incluido en el repo).
- **Docker** garantiza que la API y el dashboard corren de manera idéntica en cualquier máquina, sin depender de la configuración local de Python.
- **GitHub Actions** valida automáticamente con cada push que los tests pasen y que las imágenes Docker se construyan correctamente.

---

## Resultados del modelo (con split estratificado)

| Modelo | Accuracy | F1 (churn) | Recall (churn) | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.672 | **0.602** | **0.729** | **0.758** |
| Decision Tree | 0.635 | 0.570 | 0.712 | 0.708 |
| Random Forest | 0.707 | 0.483 | 0.403 | 0.726 |

**Modelo seleccionado: Logistic Regression** con `class_weight='balanced'` — mayor F1-score y Recall en la clase churn, que es la métrica relevante para el objetivo de negocio de AndesLink. El split estratificado garantiza que el desbalance 66/34 se mantiene consistente entre train y test.

---

## Documentación adicional

- `reports/INFORME_TECNICO_ANDESLINK_PARCIAL_1.pdf` — informe técnico de entrenamiento (EDA, preparación de datos, comparación de modelos).
- `reports/INFORME_TECNICO_ANDESLINK_PARCIAL_2.pdf` — informe técnico de despliegue (arquitectura, API, Docker, tests).
- `reports/test_report.html` — reporte detallado de los tests automatizados.

## Consideraciones de Escalabilidad (Pruebas de Rendimiento futuras):

La suite de pruebas actual con pytest valida la robustez funcional ante payloads correctos y erróneos. Para una fase posterior de producción, se prevé la incorporación de pruebas de carga stress con herramientas como Locust, orientadas a determinar el rendimiento máximo de solicitudes simultáneas (RPS) del pipeline de Machine Learning en el contenedor Docker.
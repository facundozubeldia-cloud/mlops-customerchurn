# Proyecto MLOps — AndesLink: Predicción de Churn

**Alumnos:** Facundo Zubeldia · Gonzalo Martín González Nastovich · Agustín Meza  
**Materia:** Laboratorio de Minería de Datos — ISTEA  
**Profesor:** Diego Mosquera

---

## Descripción

Este repositorio contiene el desarrollo integral de una solución de Machine Learning bajo prácticas de MLOps para la empresa ficticia **AndesLink Servicios Digitales S.A.**, que necesita anticipar el abandono de clientes (churn) para activar campañas de retención.

El proyecto cubre el ciclo completo: entrenamiento, despliegue y monitoreo. Este entregable corresponde a la **Entrega 1 — Entrenamiento**.

---

## Estructura del proyecto

```
Lab-Mineria-de-Datos/
├── data/
│   └── churn_sintetico.csv.dvc   # dataset versionado con DVC
├── models/
│   └── model_churn.pkl           # modelo serializado listo para inferencia
├── notebooks/
│   └── 01_eda_customer_churn.ipynb  # EDA, preprocesamiento y entrenamiento
├── src/
│   ├── train.py                  # script de entrenamiento reutilizable
│   └── app.py                    # dashboard Streamlit (Entrega 2)
├── reports/
│   └── informe_parcial1.pdf      # informe técnico del primer parcial
├── tests/                        # pruebas automatizadas (Entrega 2)
├── .dvc/                         # configuración DVC
├── .dvcignore
├── environment.yml               # entorno Conda reproducible
├── requirements.txt              # dependencias Python
└── README.md
```

---

## Stack tecnológico

| Herramienta | Uso |
|---|---|
| Python 3.11 | Lenguaje base |
| pandas / numpy | Procesamiento de datos |
| scikit-learn | Entrenamiento y evaluación |
| MLflow | Tracking de experimentos |
| DVC | Versionado del dataset |
| Git | Versionado del código |
| FastAPI | API de inferencia (Entrega 2) |
| Docker | Contenedorización (Entrega 2) |

---

## Instalación y configuración del entorno

### Requisitos previos
- Python 3.11
- Conda o pip
- Git
- DVC (`pip install dvc`)

### 1. Clonar el repositorio

```bash
git clone https://github.com/facundozubeldia-cloud/Lab-Mineria-de-Datos.git
cd Lab-Mineria-de-Datos
```

### 2. Crear el entorno virtual

```bash
conda env create -f environment.yml
conda activate lab-mineria
```

O con pip:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Bajar el dataset con DVC

```bash
dvc pull
```

Esto descarga `data/churn_sintetico.csv` desde el storage configurado.

---

## Ejecución

### Opción A — Notebook completo (recomendado para reproducir el análisis)

```bash
cd notebooks/
jupyter lab
```

Abrí `01_eda_customer_churn.ipynb` y ejecutá **Kernel → Restart & Run All**.

El notebook realiza en orden:
1. Carga del dataset
2. Limpieza de datos
3. Análisis exploratorio (EDA)
4. Preprocesamiento con Pipeline (StandardScaler + OneHotEncoder)
5. Comparación de 3 modelos (Logistic Regression, Decision Tree, Random Forest)
6. Selección del modelo ganador por F1-score
7. Serialización del modelo en `models/model_churn.pkl`

### Opción B — Script de entrenamiento

```bash
python src/train.py
```

Entrena el modelo y lo guarda en `models/model_churn.pkl`.

---

## Tracking de experimentos con MLflow

Los experimentos quedan registrados en `notebooks/mlflow.db`.

Para visualizarlos:

```bash
mlflow ui --backend-store-uri sqlite:////ruta/absoluta/al/proyecto/notebooks/mlflow.db --port 5001
```

Abrí `localhost:5001` → pestaña **Model training** → experimento `AndesLink_Churn_Prediction`.

---

## Estrategia de reproducibilidad

El proyecto usa **Git + DVC + MLflow** como stack de reproducibilidad:

- **Git** versiona el código fuente y los archivos de configuración.
- **DVC** versiona el dataset (`churn_sintetico.csv.dvc`). El CSV real no se commitea al repo; se almacena en un remote local (`/tmp/andeslink-dvc-storage`). En un entorno productivo este remote sería S3 o GCS.
- **MLflow** registra todos los experimentos, hiperparámetros y métricas, permitiendo comparar y reproducir cualquier run.

Esta combinación cubre el espíritu de reproducibilidad que exige la consigna para la Entrega 1.

---

## Resultados del modelo (Entrega 1)

| Modelo | Accuracy | F1 (churn) | Recall (churn) | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.703 | **0.609** | **0.704** | **0.768** |
| Decision Tree | 0.671 | 0.582 | 0.698 | 0.729 |
| Random Forest | 0.712 | 0.444 | 0.351 | 0.732 |

**Modelo seleccionado: Logistic Regression** — mayor F1-score y Recall en la clase churn, que es la métrica relevante para el objetivo de negocio de AndesLink.
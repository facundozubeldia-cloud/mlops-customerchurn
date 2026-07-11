"""
Script de monitoreo con Evidently — AndesLink Churn Prediction
Genera un reporte HTML comparando el dataset de referencia (entrenamiento)
contra datos de producción simulados para detectar drift.

Correr con: python scripts/generar_reporte_evidently.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently import Report

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "churn_sintetico.csv"
REPORT_PATH = ROOT / "reports" / "reporte_evidently.html"

# ── Cargar dataset de referencia ───────────────────────────────────────────────
print("Cargando dataset de referencia...")
df = pd.read_csv(DATA_PATH)

# Separar features de la variable objetivo
features = [c for c in df.columns if c != "churn"]
df_ref = df[features].copy()

# ── Simular datos de producción con drift ──────────────────────────────────────
print("Simulando datos de producción con drift...")
np.random.seed(99)
n = 1000

df_prod = pd.DataFrame({
    # Drift: clientes más nuevos en producción (tenure más bajo)
    "tenure_months": np.random.randint(1, 20, n),
    # Drift: cargos más altos en producción
    "monthly_charge": np.random.uniform(80, 150, n),
    "total_charges": np.random.uniform(80, 3000, n),
    # Drift: más tickets de soporte
    "support_tickets": np.random.randint(2, 10, n),
    "late_payments": np.random.randint(0, 5, n),
    "avg_monthly_usage_gb": np.random.uniform(0, 300, n),
    # Drift: predominan contratos mensuales
    "contract_type": np.random.choice(["mensual", "anual", "bianual"], n, p=[0.75, 0.15, 0.10]),
    "payment_method": np.random.choice(["debito", "credito", "transferencia", "efectivo"], n),
    "internet_service": np.random.choice(["fibra", "cable", "movil", "ninguno"], n),
    "region": np.random.choice(["centro", "norte", "sur", "oeste"], n),
    "has_streaming": np.random.randint(0, 2, n),
    "has_security_pack": np.random.randint(0, 2, n),
    "num_products": np.random.randint(1, 5, n),
    "customer_age": np.random.randint(18, 80, n),
    "is_promo": np.random.randint(0, 2, n),
})

# ── Definir estructura de datos para Evidently ─────────────────────────────────
print("Configurando Evidently...")

cat_cols = ["contract_type", "payment_method", "internet_service", "region"]
num_cols = [c for c in features if c not in cat_cols]

data_def = DataDefinition(
    numerical_columns=num_cols,
    categorical_columns=cat_cols,
)

# ── Crear datasets de Evidently ────────────────────────────────────────────────
ref_dataset  = Dataset.from_pandas(df_ref,  data_definition=data_def)
prod_dataset = Dataset.from_pandas(df_prod, data_definition=data_def)

# ── Generar reporte ────────────────────────────────────────────────────────────
print("Generando reporte...")
report = Report(metrics=[
    DataSummaryPreset(),
    DataDriftPreset(),
])

my_eval = report.run(reference_data=ref_dataset, current_data=prod_dataset)
my_eval.save_html(str(REPORT_PATH))

print(f"Reporte generado en: {REPORT_PATH}")
print("Abrilo en el navegador para ver el análisis de drift.")
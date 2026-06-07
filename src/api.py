from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd

# ── Cargar modelo ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_churn.pkl"

try:
    modelo = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. Corré train.py primero.")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AndesLink Churn Predictor API",
    description="API de inferencia para predicción de abandono de clientes.",
    version="1.0.0"
)

# ── Schema de entrada ──────────────────────────────────────────────────────────
class ClienteInput(BaseModel):
    tenure_months: int = Field(..., ge=0, example=12)
    monthly_charge: float = Field(..., ge=0, example=65.0)
    total_charges: float = Field(..., ge=0, example=780.0)
    support_tickets: int = Field(..., ge=0, example=2)
    late_payments: int = Field(..., ge=0, example=1)
    avg_monthly_usage_gb: float = Field(..., ge=0, example=120.0)
    contract_type: str = Field(..., example="mensual")
    payment_method: str = Field(..., example="debito")
    internet_service: str = Field(..., example="fibra")
    region: str = Field(..., example="centro")
    has_streaming: int = Field(..., ge=0, le=1, example=1)
    has_security_pack: int = Field(..., ge=0, le=1, example=0)
    num_products: int = Field(..., ge=1, example=2)
    customer_age: int = Field(..., ge=18, example=35)
    is_promo: int = Field(..., ge=0, le=1, example=0)

# ── Schema de salida ───────────────────────────────────────────────────────────
class PrediccionOutput(BaseModel):
    churn: int
    probabilidad_churn: float
    riesgo: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "modelo": "Logistic Regression", "version": "1.0.0"}

@app.post("/predict", response_model=PrediccionOutput)
def predecir(cliente: ClienteInput):
    try:
        # Convertir input a DataFrame con las columnas en el orden correcto
        datos = pd.DataFrame([cliente.model_dump()])

        # Predicción
        churn = int(modelo.predict(datos)[0])
        prob = float(modelo.predict_proba(datos)[0][1])

        # Nivel de riesgo
        if prob >= 0.7:
            riesgo = "alto"
        elif prob >= 0.4:
            riesgo = "medio"
        else:
            riesgo = "bajo"

        return PrediccionOutput(
            churn=churn,
            probabilidad_churn=round(prob, 4),
            riesgo=riesgo
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
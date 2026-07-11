import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Cargar modelo ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_churn.pkl"

modelo = None
try:
    modelo = joblib.load(MODEL_PATH)
    logger.info(f"Modelo cargado correctamente desde {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Modelo no encontrado en {MODEL_PATH}. Corré train.py primero.")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AndesLink Churn Predictor API",
    description="API de inferencia para predicción de abandono de clientes.",
    version="1.0.0"
)
#instrumentación
Instrumentator().instrument(app).expose(app)
# ── Schema de entrada ──────────────────────────────────────────────────────────
class ClienteInput(BaseModel):
    tenure_months:        int   = Field(..., ge=0,  json_schema_extra={"example": 12})
    monthly_charge:       float = Field(..., ge=0,  json_schema_extra={"example": 65.0})
    total_charges:        float = Field(..., ge=0,  json_schema_extra={"example": 780.0})
    support_tickets:      int   = Field(..., ge=0,  json_schema_extra={"example": 2})
    late_payments:        int   = Field(..., ge=0,  json_schema_extra={"example": 1})
    avg_monthly_usage_gb: float = Field(..., ge=0,  json_schema_extra={"example": 120.0})
    contract_type:        Literal["mensual", "anual", "bianual"]                          = Field(..., json_schema_extra={"example": "mensual"})
    payment_method:       Literal["debito", "credito", "transferencia", "efectivo"]       = Field(..., json_schema_extra={"example": "debito"})
    internet_service:     Literal["fibra", "cable", "movil", "ninguno"]                   = Field(..., json_schema_extra={"example": "fibra"})
    region:               Literal["centro", "norte", "sur", "oeste"]                      = Field(..., json_schema_extra={"example": "centro"})
    has_streaming:        int   = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    has_security_pack:    int   = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    num_products:         int   = Field(..., ge=1,       json_schema_extra={"example": 2})
    customer_age:         int   = Field(..., ge=18,      json_schema_extra={"example": 35})
    is_promo:             int   = Field(..., ge=0, le=1, json_schema_extra={"example": 0})

# ── Schema de salida ───────────────────────────────────────────────────────────
class PrediccionOutput(BaseModel):
    churn: int
    probabilidad_churn: float
    riesgo: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "modelo": "Logistic Regression", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Healthcheck real: verifica que el modelo está cargado y puede inferir."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        datos_prueba = pd.DataFrame([{
            "tenure_months": 12, "monthly_charge": 65.0, "total_charges": 780.0,
            "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 120.0,
            "contract_type": "mensual", "payment_method": "debito",
            "internet_service": "fibra", "region": "centro",
            "has_streaming": 0, "has_security_pack": 0,
            "num_products": 2, "customer_age": 35, "is_promo": 0
        }])
        modelo.predict(datos_prueba)
        return {"status": "ok", "modelo": "cargado", "inferencia": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error en inferencia de prueba: {str(e)}")


@app.post("/predict", response_model=PrediccionOutput)
def predecir(cliente: ClienteInput):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        datos = pd.DataFrame([cliente.model_dump()])
        churn = int(modelo.predict(datos)[0])
        prob  = float(modelo.predict_proba(datos)[0][1])

        if prob >= 0.7:
            riesgo = "alto"
        elif prob >= 0.4:
            riesgo = "medio"
        else:
            riesgo = "bajo"

        logger.info(f"Prediccion — churn={churn}, prob={prob:.4f}, riesgo={riesgo}")

        return PrediccionOutput(
            churn=churn,
            probabilidad_churn=round(prob, 4),
            riesgo=riesgo
        )
    except Exception as e:
        logger.error(f"Error en prediccion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
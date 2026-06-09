"""
Tests de la API de predicción de churn — AndesLink
Correr con: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ── Payload válido de ejemplo ──────────────────────────────────────────────────
CLIENTE_VALIDO = {
    "tenure_months": 6,
    "monthly_charge": 85.0,
    "total_charges": 510.0,
    "support_tickets": 4,
    "late_payments": 2,
    "avg_monthly_usage_gb": 95.0,
    "contract_type": "mensual",
    "payment_method": "efectivo",
    "internet_service": "movil",
    "region": "centro",
    "has_streaming": 0,
    "has_security_pack": 0,
    "num_products": 1,
    "customer_age": 28,
    "is_promo": 1
}


# ── Test 1: Health check ───────────────────────────────────────────────────────
def test_health_check():
    """El endpoint raíz debe responder 200 con status ok."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelo" in data


# ── Test 2: Predicción válida ──────────────────────────────────────────────────
def test_prediccion_cliente_valido():
    """Una predicción con datos válidos debe devolver 200 y campos correctos."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probabilidad_churn" in data
    assert "riesgo" in data
    assert data["churn"] in [0, 1]
    assert 0.0 <= data["probabilidad_churn"] <= 1.0
    assert data["riesgo"] in ["bajo", "medio", "alto"]


# ── Test 3: Validación de campos faltantes ─────────────────────────────────────
def test_prediccion_payload_incompleto():
    """Un payload sin campos obligatorios debe devolver 422."""
    payload_incompleto = {"tenure_months": 12}
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422


# ── Test 4: Validación de tipos incorrectos ────────────────────────────────────
def test_prediccion_tipo_incorrecto():
    """Un payload con tipos incorrectos debe devolver 422."""
    payload_malo = CLIENTE_VALIDO.copy()
    payload_malo["tenure_months"] = "doce"  # debe ser int
    response = client.post("/predict", json=payload_malo)
    assert response.status_code == 422


# ── Test 5: Cliente de bajo riesgo ────────────────────────────────────────────
def test_prediccion_cliente_bajo_riesgo():
    """Un cliente con perfil de bajo riesgo debe tener probabilidad < 0.7."""
    cliente_fiel = {
        "tenure_months": 60,
        "monthly_charge": 45.0,
        "total_charges": 2700.0,
        "support_tickets": 0,
        "late_payments": 0,
        "avg_monthly_usage_gb": 150.0,
        "contract_type": "bianual",
        "payment_method": "debito",
        "internet_service": "fibra",
        "region": "norte",
        "has_streaming": 1,
        "has_security_pack": 1,
        "num_products": 4,
        "customer_age": 45,
        "is_promo": 0
    }
    response = client.post("/predict", json=cliente_fiel)
    assert response.status_code == 200
    data = response.json()
    assert data["probabilidad_churn"] < 0.7
